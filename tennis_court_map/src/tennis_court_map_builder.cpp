#include "tennis_court_map/tennis_court_map_builder.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tennis_court_map {
namespace {

int metersToCells(double meters, double resolution) {
    return static_cast<int>(std::ceil(meters / resolution));
}

int worldXToIndex(double x, double origin_x, double resolution, int width) {
    const int ix = static_cast<int>(std::floor((x - origin_x) / resolution));
    return std::clamp(ix, 0, width - 1);
}

int worldYToIndex(double y, double origin_y, double resolution, int height) {
    const int iy = static_cast<int>(std::floor((y - origin_y) / resolution));
    return std::clamp(iy, 0, height - 1);
}

void setCell(std::vector<int8_t>& data, int width, int height, int ix, int iy, int8_t value) {
    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
        return;
    }
    data[static_cast<size_t>(iy) * static_cast<size_t>(width) + static_cast<size_t>(ix)] = value;
}

void drawHorizontalLine(std::vector<int8_t>& data, int width, int height, double origin_x,
                        double origin_y, double resolution, double y, double x0, double x1,
                        double line_width, int8_t value) {
    const double y_start = y - line_width * 0.5;
    const double y_end = y + line_width * 0.5;
    const int iy0 = worldYToIndex(y_start, origin_y, resolution, height);
    const int iy1 = worldYToIndex(y_end, origin_y, resolution, height);
    const int ix0 = worldXToIndex(std::min(x0, x1), origin_x, resolution, width);
    const int ix1 = worldXToIndex(std::max(x0, x1), origin_x, resolution, width);

    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            setCell(data, width, height, ix, iy, value);
        }
    }
}

void drawVerticalLine(std::vector<int8_t>& data, int width, int height, double origin_x,
                      double origin_y, double resolution, double x, double y0, double y1,
                      double line_width, int8_t value) {
    const double x_start = x - line_width * 0.5;
    const double x_end = x + line_width * 0.5;
    const int ix0 = worldXToIndex(x_start, origin_x, resolution, width);
    const int ix1 = worldXToIndex(x_end, origin_x, resolution, width);
    const int iy0 = worldYToIndex(std::min(y0, y1), origin_y, resolution, height);
    const int iy1 = worldYToIndex(std::max(y0, y1), origin_y, resolution, height);

    for (int ix = ix0; ix <= ix1; ++ix) {
        for (int iy = iy0; iy <= iy1; ++iy) {
            setCell(data, width, height, ix, iy, value);
        }
    }
}

void fillRectangle(std::vector<int8_t>& data, int width, int height, double origin_x,
                   double origin_y, double resolution, double x0, double y0, double x1,
                   double y1, int8_t value) {
    const int ix0 = worldXToIndex(std::min(x0, x1), origin_x, resolution, width);
    const int ix1 = worldXToIndex(std::max(x0, x1), origin_x, resolution, width);
    const int iy0 = worldYToIndex(std::min(y0, y1), origin_y, resolution, height);
    const int iy1 = worldYToIndex(std::max(y0, y1), origin_y, resolution, height);

    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            setCell(data, width, height, ix, iy, value);
        }
    }
}

} // namespace

nav_msgs::msg::OccupancyGrid buildTennisCourtMap(const TennisCourtMapOptions& options) {
    const double margin = options.margin_m;
    const double resolution = options.resolution;

    // 球场长边沿 map 的 x 轴，短边沿 y 轴；球场中心在世界原点
    const double map_width_m = TennisCourtSpec::kCourtLength + 2.0 * margin;
    const double map_height_m = TennisCourtSpec::kDoublesWidth + 2.0 * margin;

    const int width = metersToCells(map_width_m, resolution);
    const int height = metersToCells(map_height_m, resolution);

    nav_msgs::msg::OccupancyGrid grid;
    grid.header.frame_id = options.frame_id;
    grid.info.resolution = static_cast<float>(resolution);
    grid.info.width = static_cast<uint32_t>(width);
    grid.info.height = static_cast<uint32_t>(height);

    const double origin_x = -TennisCourtSpec::kCourtLength * 0.5 - margin;
    const double origin_y = -TennisCourtSpec::kDoublesWidth * 0.5 - margin;
    grid.info.origin.position.x = origin_x;
    grid.info.origin.position.y = origin_y;
    grid.info.origin.position.z = 0.0;
    grid.info.origin.orientation.w = 1.0;

    std::vector<int8_t> data(static_cast<size_t>(width) * static_cast<size_t>(height), 100);

    const double court_x0 = -TennisCourtSpec::kCourtLength * 0.5;
    const double court_x1 = TennisCourtSpec::kCourtLength * 0.5;
    const double court_y0 = -TennisCourtSpec::kDoublesWidth * 0.5;
    const double court_y1 = TennisCourtSpec::kDoublesWidth * 0.5;

    const double singles_y0 = -TennisCourtSpec::kSinglesWidth * 0.5;
    const double singles_y1 = TennisCourtSpec::kSinglesWidth * 0.5;

    const double net_x = 0.0;
    const double service_near_x = -TennisCourtSpec::kServiceLineFromNet;
    const double service_far_x = TennisCourtSpec::kServiceLineFromNet;
    const double center_y = 0.0;

    // 场内可通行区域
    fillRectangle(data, width, height, origin_x, origin_y, resolution, court_x0, court_y0,
                  court_x1, court_y1, 0);

    const double lw = TennisCourtSpec::kLineWidth;
    constexpr int8_t kLine = 100;

    // 底线（x 方向两端）
    drawVerticalLine(data, width, height, origin_x, origin_y, resolution, court_x0, court_y0,
                     court_y1, lw, kLine);
    drawVerticalLine(data, width, height, origin_x, origin_y, resolution, court_x1, court_y0,
                     court_y1, lw, kLine);

    // 双打边线（y 方向两侧）
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, court_y0, court_x0,
                       court_x1, lw, kLine);
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, court_y1, court_x0,
                       court_x1, lw, kLine);

    // 单打边线（全场）
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, singles_y0, court_x0,
                       court_x1, lw, kLine);
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, singles_y1, court_x0,
                       court_x1, lw, kLine);

    // 发球线（垂直于长边，即 x 方向）
    drawVerticalLine(data, width, height, origin_x, origin_y, resolution, service_near_x,
                     court_y0, court_y1, lw, kLine);
    drawVerticalLine(data, width, height, origin_x, origin_y, resolution, service_far_x,
                     court_y0, court_y1, lw, kLine);

    // 中线（发球区，沿 x 轴）
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, center_y,
                       service_near_x, net_x, lw, kLine);
    drawHorizontalLine(data, width, height, origin_x, origin_y, resolution, center_y, net_x,
                       service_far_x, lw, kLine);

    // 球网（过原点，x = 0）
    drawVerticalLine(data, width, height, origin_x, origin_y, resolution, net_x, court_y0,
                     court_y1, lw, kLine);

    grid.data = std::move(data);
    return grid;
}

} // namespace tennis_court_map
