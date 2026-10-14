#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace loop_retrieval {

// Ages and separation are measured in pose-graph keyframes, after input thinning.
using Candidate = std::pair<int, float>;

inline std::vector<Candidate> selectTemporalCandidates(
    std::vector<Candidate> candidates, int frame_index, int min_keyframe_gap,
    int remote_keyframe_gap, int candidate_separation,
    double best_score_threshold, double candidate_score_threshold,
    double relative_ratio, std::size_t max_candidates) {
  if (max_candidates == 0 || remote_keyframe_gap <= 0 || candidate_separation < 0) {
    return {};
  }
  const auto by_score = [](const Candidate &a, const Candidate &b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  };
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
      [&](const Candidate &candidate) {
        return candidate.first < 0 || candidate.first >= frame_index - min_keyframe_gap ||
               !std::isfinite(candidate.second) ||
               candidate.second <= candidate_score_threshold;
      }), candidates.end());
  std::sort(candidates.begin(), candidates.end(), by_score);

  std::vector<Candidate> recent;
  std::vector<Candidate> remote;
  for (const auto &candidate : candidates) {
    auto &bucket = candidate.first < frame_index - remote_keyframe_gap ? remote : recent;
    if (std::none_of(bucket.begin(), bucket.end(), [&](const Candidate &kept) {
          return kept.first == candidate.first;
        })) {
      bucket.push_back(candidate);
    }
  }
  const auto filter_bucket = [&](std::vector<Candidate> &bucket) {
    if (bucket.empty()) {
      return;
    }
    const double best_score = bucket.front().second;
    if (best_score <= best_score_threshold) {
      bucket.clear();
      return;
    }
    bucket.erase(std::remove_if(bucket.begin(), bucket.end(), [&](const Candidate &candidate) {
      return candidate.second < best_score * relative_ratio;
    }), bucket.end());
  };
  filter_bucket(recent);
  filter_bucket(remote);

  std::vector<Candidate> selected;
  // Verify one qualified remote result first: the caller stops at its first
  // accepted loop, so reserving only the last slot could still starve history.
  if (!remote.empty()) {
    selected.push_back(remote.front());
  }
  std::vector<Candidate> ranked = recent;
  ranked.insert(ranked.end(), remote.begin(), remote.end());
  std::sort(ranked.begin(), ranked.end(), by_score);
  const auto append_candidates = [&](bool require_separation) {
    for (const auto &candidate : ranked) {
      if (selected.size() >= max_candidates) {
        break;
      }
      const bool conflicts = std::any_of(selected.begin(), selected.end(),
          [&](const Candidate &kept) {
            return candidate.first == kept.first ||
                   (require_separation &&
                    std::abs(candidate.first - kept.first) < candidate_separation);
          });
      if (!conflicts) {
        selected.push_back(candidate);
      }
    }
  };
  append_candidates(true);
  // With only one historical neighborhood, use the remaining budget for
  // alternate views rather than discarding all its other geometric chances.
  append_candidates(false);
  return selected;
}

}  // namespace loop_retrieval
