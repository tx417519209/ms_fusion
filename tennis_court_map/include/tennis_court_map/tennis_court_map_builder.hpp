#ifndef TENNIS_COURT_MAP_TENNIS_COURT_MAP_BUILDER_HPP
#define TENNIS_COURT_MAP_TENNIS_COURT_MAP_BUILDER_HPP

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <string>

namespace tennis_court_map {

// ITF 标准硬地/草地双打球场尺寸（米）
struct TennisCourtSpec {
    static constexpr double kCourtLength = 23.77;
    static constexpr double kDoublesWidth = 10.97;
    static constexpr double kSinglesWidth = 8.23;
    static constexpr double kServiceLineFromNet = 6.40;
    static constexpr double kLineWidth = 0.05;
};

struct TennisCourtMapOptions {
    double resolution = 0.05;
    double margin_m = 3.0;
    std::string frame_id = "world";
};

nav_msgs::msg::OccupancyGrid buildTennisCourtMap(const TennisCourtMapOptions& options);

} // namespace tennis_court_map

#endif // TENNIS_COURT_MAP_TENNIS_COURT_MAP_BUILDER_HPP
