#ifndef MSF_CORE_INTERFACE_EIGEN_UTILS_HPP
#define MSF_CORE_INTERFACE_EIGEN_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Eigen的数据对齐
namespace Eigen {

template <typename T>
using aligned_list = std::list<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;
        
template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                             Eigen::aligned_allocator<std::pair<K const, V>>>;
        
template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                                                 Eigen::aligned_allocator<std::pair<K const, V>>>;

} // namespace Eigen

#endif // MSF_CORE_INTERFACE_EIGEN_UTILS_HPP