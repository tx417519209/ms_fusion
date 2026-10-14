#!/usr/bin/env bash
# Download one or more TUM-VI EuRoC 512x512 exports with safe resume support.
#
# Examples:
#   bash dataset/download_tumvi_dataset.sh corridor1
#   bash dataset/download_tumvi_dataset.sh outdoors4 slides1
#   bash dataset/download_tumvi_dataset.sh room1 corridor1 outdoors4

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <tumvi-sequence> [<tumvi-sequence> ...]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "$0")/.." && pwd)
base_url="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"

for sequence in "$@"; do
  if [[ ! "$sequence" =~ ^[a-z0-9]+$ ]]; then
    echo "invalid TUM-VI sequence name: $sequence" >&2
    exit 2
  fi

  output_dir="$repo_root/dataset/tumvi_$sequence"
  archive="dataset-${sequence}_512_16.tar"
  mkdir -p "$output_dir"
  curl -fL --retry 3 --continue-at - \
    -o "$output_dir/$archive" \
    "$base_url/$archive"
done
