#!/usr/bin/env python3
"""Cluster a DBoW3-compatible vocabulary locally (CPU, sklearn KMeans) from
descriptors dumped on the board by train_xfeat_voc --features_out.

Why: DBoW3::Vocabulary::create()'s own from-scratch k-means (C++,
ThirdParty/DBoW3) has no iteration cap and can stall/oscillate for many
minutes on deep tree levels when a node's descriptor count is small relative
to the branching factor k -- exactly the regime you hit once the tree goes a
few levels deep. This script does the same hierarchical k-means with
sklearn (vectorized, well-behaved on small clusters, much faster on a normal
dev machine) and writes a .dbow3 file byte-for-byte compatible with this
repo's vendored DBoW3 loader (ros_wrapper/src/ms_fusion_loop/src/ThirdParty/
DBoW3/src/Vocabulary.cpp: Vocabulary::toStream/fromStream).

Board side (extract only, no clustering):
    ros2 run ms_fusion_loop train_xfeat_voc -- \\
        --image_dir <images> --model <xfeat.axmodel> \\
        --features_out outdoors4_features.msfv

Local side (this script):
    python3 scripts/dbow3_train_from_features.py \\
        --features outdoors4_features.msfv \\
        --output ros_wrapper/src/ms_fusion_loop/voc/xfeat_voc_outdoors4.dbow3 \\
        --k 10 --L 4

Output format notes (see Vocabulary::toStream, compressed=False branch):
  the file is a small header + an UNCOMPRESSED dump of the tree -- this
  script never touches DBoW3's quicklz compression, it just writes
  compressed=0 and the same plain layout toStream() would put in aux_stream.
  DBoW3::Vocabulary::load() reads either form transparently.

Requires: numpy, scikit-learn (pip install numpy scikit-learn).
"""

import argparse
import math
import struct
import sys
import time

import numpy as np

MSFV_MAGIC = 0x4D534656  # 'MSFV', matches saveFeatures() in tools/train_voc_common.cpp
DBOW3_SIG = 88877711233  # matches Vocabulary::toStream's magic number
CV_32FC1 = 5  # opencv type constant for cv::Mat(..., CV_32FC1)

# DBoW3::WeightingType / ScoringType enum values (BowVector.h)
TF_IDF = 0
L1_NORM = 0


def load_features(path):
    """Read the .msfv dump written by saveFeatures() in train_voc_common.cpp."""
    with open(path, "rb") as f:
        magic, dim, num_images = struct.unpack("<III", f.read(12))
        if magic != MSFV_MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}, expected 0x{MSFV_MAGIC:08x}")
        images = []
        for _ in range(num_images):
            (rows,) = struct.unpack("<I", f.read(4))
            if rows == 0:
                images.append(np.zeros((0, dim), dtype=np.float32))
                continue
            buf = f.read(4 * rows * dim)
            arr = np.frombuffer(buf, dtype="<f4").reshape(rows, dim)
            images.append(arr.copy())
    return images, dim


class Node:
    __slots__ = ("id", "parent", "children", "descriptor", "weight", "word_id")

    def __init__(self, node_id, parent):
        self.id = node_id
        self.parent = parent
        self.children = []
        self.descriptor = None  # np.float32[dim]
        self.weight = 0.0
        self.word_id = None


def build_tree(all_descs, k, L, min_leaf_for_kmeans=1, verbose=True):
    """Hierarchical k-means over all_descs (N x D float32), mirroring
    DBoW3::Vocabulary::HKmeansStep. Returns (nodes, leaf_desc_indices) where
    nodes[0] is the (descriptor-less) root and leaf_desc_indices maps each
    leaf Node.id -> np.ndarray of global descriptor row indices assigned to it.
    """
    from sklearn.cluster import KMeans

    nodes = [Node(0, 0)]  # root; never written to the file
    leaf_desc_indices = {}
    next_id = [1]
    t0 = time.time()
    node_count = [0]

    def split(parent_id, indices, level):
        if indices.size == 0:
            return
        node_count[0] += 1
        if node_count[0] % 200 == 0 and verbose:
            elapsed = time.time() - t0
            print(f"[dbow3_train] ...{node_count[0]} nodes built, "
                  f"{len(nodes)} total so far, {elapsed:.1f}s elapsed", file=sys.stderr)

        if indices.size <= k:
            # trivial case: one cluster per point (same as DBoW3's HKmeansStep)
            groups = [np.array([i]) for i in indices]
            centers = all_descs[indices]
        else:
            km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=0)
            labels = km.fit_predict(all_descs[indices])
            centers = km.cluster_centers_.astype(np.float32)
            groups = [indices[labels == c] for c in range(k)]

        for c in range(len(centers)):
            child_indices = groups[c]
            if child_indices.size == 0:
                continue  # degenerate empty cluster: skip, matches negligible-impact case
            nid = next_id[0]
            next_id[0] += 1
            node = Node(nid, parent_id)
            node.descriptor = np.asarray(centers[c], dtype=np.float32)
            nodes.append(node)
            nodes[parent_id].children.append(nid)

            if level < L and child_indices.size > 1:
                split(nid, child_indices, level + 1)
            else:
                leaf_desc_indices[nid] = child_indices

    split(0, np.arange(all_descs.shape[0]), 1)
    return nodes, leaf_desc_indices


def assign_word_ids(nodes, leaf_desc_indices):
    """Mirrors Vocabulary::createWords(): word_id = position among leaves,
    in ascending node-id order (== the order nodes were created, skipping root)."""
    words = []  # list of node ids, in word_id order
    for node in nodes[1:]:
        if not node.children:  # isLeaf()
            node.word_id = len(words)
            words.append(node.id)
    return words


def compute_weights(nodes, leaf_desc_indices, words, image_of_descriptor, num_images):
    """TF_IDF: weight[word] = ln(NDocs / Ni[word]), Ni = #images containing >=1
    descriptor assigned to that leaf. Mirrors Vocabulary::setNodeWeights()."""
    for nid, indices in leaf_desc_indices.items():
        img_ids = image_of_descriptor[indices]
        ni = np.unique(img_ids).size
        if ni > 0:
            nodes[nid].weight = math.log(float(num_images) / float(ni))
        # else: leave weight at its default 0.0 (cannot occur with a non-empty leaf)


def write_dbow3(path, nodes, k, L, words):
    with open(path, "wb") as f:
        nnodes = len(nodes)
        f.write(struct.pack("<Q", DBOW3_SIG))
        f.write(struct.pack("<?", False))  # compressed = false
        f.write(struct.pack("<I", nnodes))

        # aux_stream content, written directly since compressed=False
        f.write(struct.pack("<i", k))
        f.write(struct.pack("<i", L))
        f.write(struct.pack("<i", L1_NORM))  # m_scoring
        f.write(struct.pack("<i", TF_IDF))  # m_weighting

        for node in nodes[1:]:  # root (id 0) is implicit, never written
            f.write(struct.pack("<I", node.id))
            f.write(struct.pack("<I", node.parent))
            f.write(struct.pack("<d", node.weight))
            desc = node.descriptor
            cols = int(desc.shape[0])
            f.write(struct.pack("<i", cols))  # DescManip::toStream: cols
            f.write(struct.pack("<i", 1))     # rows
            f.write(struct.pack("<i", CV_32FC1))  # type
            f.write(np.ascontiguousarray(desc, dtype="<f4").tobytes())

        f.write(struct.pack("<I", len(words)))
        for word_id, node_id in enumerate(words):
            f.write(struct.pack("<I", word_id))
            f.write(struct.pack("<I", node_id))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help=".msfv dump from train_xfeat_voc --features_out")
    ap.add_argument("--output", required=True, help="output .dbow3 path")
    ap.add_argument("--k", type=int, default=10, help="branching factor (default 10)")
    ap.add_argument("--L", type=int, default=4,
                     help="tree depth (default 4 -> k^L=10000 words; 5 needs a lot more "
                          "training descriptors per the corridor1/outdoors4 stall issue)")
    args = ap.parse_args()

    print(f"[dbow3_train] loading {args.features} ...")
    images, dim = load_features(args.features)
    num_images = len(images)
    total_descs = sum(im.shape[0] for im in images)
    print(f"[dbow3_train] images={num_images} dim={dim} total_descriptors={total_descs} "
          f"k={args.k} L={args.L} (target words=k^L={args.k ** args.L})")
    if total_descs < args.k ** args.L * 5:
        print(f"[dbow3_train] WARNING: only ~{total_descs / max(1, args.k ** args.L):.1f} "
              f"descriptors per leaf word on average; consider a smaller --L or more "
              f"training images if retrieval quality looks poor.", file=sys.stderr)

    all_descs = np.concatenate([im for im in images if im.shape[0] > 0], axis=0)
    image_of_descriptor = np.concatenate([
        np.full(im.shape[0], i, dtype=np.int32) for i, im in enumerate(images) if im.shape[0] > 0
    ])

    print("[dbow3_train] clustering (hierarchical k-means, sklearn)...")
    t0 = time.time()
    nodes, leaf_desc_indices = build_tree(all_descs, args.k, args.L)
    words = assign_word_ids(nodes, leaf_desc_indices)
    compute_weights(nodes, leaf_desc_indices, words, image_of_descriptor, num_images)
    print(f"[dbow3_train] done clustering: nodes={len(nodes)} words={len(words)} "
          f"({time.time() - t0:.1f}s)")

    write_dbow3(args.output, nodes, args.k, args.L, words)
    print(f"[dbow3_train] wrote {args.output}")


if __name__ == "__main__":
    main()
