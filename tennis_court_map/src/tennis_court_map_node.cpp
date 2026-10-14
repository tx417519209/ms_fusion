#include "tennis_court_map/tennis_court_map_builder.hpp"

#include <rclcpp/rclcpp.hpp>

namespace tennis_court_map {

class TennisCourtMapNode : public rclcpp::Node {
public:
    TennisCourtMapNode() : Node("tennis_court_map") {
        frame_id_ = declare_parameter<std::string>("frame_id", "world");
        map_topic_ = declare_parameter<std::string>("map_topic", "tennis_court_map");
        resolution_ = declare_parameter<double>("resolution", 0.05);
        margin_m_ = declare_parameter<double>("margin_m", 3.0);
        publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 1.0);

        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.transient_local();
        qos.reliable();

        publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic_, qos);

        TennisCourtMapOptions options;
        options.frame_id = frame_id_;
        options.resolution = resolution_;
        options.margin_m = margin_m_;
        map_msg_ = buildTennisCourtMap(options);

        const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 0.1));
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(period),
            std::bind(&TennisCourtMapNode::publishMap, this));

        RCLCPP_INFO(get_logger(),
                    "tennis court map ready: topic=%s frame=%s size=%ux%u resolution=%.3f margin=%.1fm",
                    map_topic_.c_str(), frame_id_.c_str(), map_msg_.info.width, map_msg_.info.height,
                    resolution_, margin_m_);
        RCLCPP_INFO(get_logger(),
                    "court (ITF doubles): length=%.2fm along +x, width=%.2fm along +y, center at (0,0)",
                    TennisCourtSpec::kCourtLength, TennisCourtSpec::kDoublesWidth);
        RCLCPP_INFO(get_logger(),
                    "map origin: (%.2f, %.2f), size: %.2fm x %.2fm",
                    map_msg_.info.origin.position.x, map_msg_.info.origin.position.y,
                    map_msg_.info.width * map_msg_.info.resolution,
                    map_msg_.info.height * map_msg_.info.resolution);
        publishMap();
    }

private:
    void publishMap() {
        map_msg_.header.stamp = now();
        publisher_->publish(map_msg_);
    }

    std::string frame_id_;
    std::string map_topic_;
    double resolution_{0.05};
    double margin_m_{3.0};
    double publish_rate_hz_{1.0};

    nav_msgs::msg::OccupancyGrid map_msg_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace tennis_court_map

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<tennis_court_map::TennisCourtMapNode>());
    rclcpp::shutdown();
    return 0;
}
