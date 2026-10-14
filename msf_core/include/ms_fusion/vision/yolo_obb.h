#ifndef MS_FUSION_VISION_YOLO_OBB_H
#define MS_FUSION_VISION_YOLO_OBB_H

#include "config_params.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/measurement.h"

#include "infer_api.h"

#include <array>

namespace ms_fusion {

class GenericCamera;

struct YoloFrameTiming {
    double debug_copy_ms = -1.0;
    double preprocess_ms = -1.0;
    double bind_ms = -1.0;
    // infer_api::run_model includes AX inference plus SDK output decode/NMS.
    double infer_ms = -1.0;
    // Fetching the already-decoded result reference (normally near zero).
    double output_ms = -1.0;
    // ms_fusion-side OBB -> line/keypoint geometry conversion.
    double post_ms = -1.0;
    double release_ms = -1.0;
    double total_ms = -1.0;
    int detections_left = -1;
    int detections_right = -1;
    int lines_left = -1;
    int lines_right = -1;
    int keypoints_left = -1;
    int keypoints_right = -1;
};

// YOLO类
class YOLO {
public:
    // 构造函数
    YOLO(const RelocConfigParams& reloc_config_params, 
         const std::vector<std::shared_ptr<GenericCamera>>& cameras);

    // 接口
    void process(const double& timestamp, const cv::Mat& image0, const cv::Mat& image1);

    // 获取重定位结果
    void info(double& timestamp, ReLocFeatInfo& reloc_feat_info);

    // 获取上一帧 YOLO-OBB 可视化数据
    const RelocDebugInfo& getDebugInfo() const { return debug_info_; }

    const YoloFrameTiming& lastFrameTiming() const { return last_frame_timing_; }

private:
    /**
     * 进行后处理
     * @param 网络检测的结果
     * @return 后处理结果
     */
    void postProcess(const std::pair<std::vector<ax_yolo_obb_infer_node::obb_detection>, 
                      std::vector<ax_yolo_obb_infer_node::obb_detection>>& pair_dets);

    /**
     * obb框转到线
     * @param obb框
     * @param 线
     */
    void obbToLine(const std::vector<ax_yolo_obb_infer_node::obb_detection>& dets,
                   std::vector<std::vector<ObbLine>>& obb_to_lines);

    /**
     * 寻找NET点和POST点
     */
    void findNetAndPostPoints(const std::vector<std::vector<ObbLine>>& obb_to_lines,
                              std::vector<int>& feature_ids, std::vector<cv::Point2f>& feature_pixels);

    /**
     * 线的ID与特征点的ID映射
     */
    int lineIdToNetFeatureId(const int& line_id);

    /**
    * 线的ID映射到POST特征点的ID
    * @param 线的ID
    * @return POST特征点的ID
    */
    std::vector<int> lineIdToPostFeatureIds(const int& line_id);

    /**
     * 通过线的点，求线的一般表达时
     * @param 线的点
     * @param 线的一般表达式
     */
    void segmentLinesToLines(const std::vector<std::vector<ObbLine>>& obb_to_lines, 
                             std::vector<Line>& lines);

    /**
     * 通过线的点，求线的一般表达时
     * @param 线的点
     * @param 线的一般表达式
     */
    cv::Vec3d line2dFromSegment(const cv::Point2f& start_point, const cv::Point2f& end_point);

    /**
    * 根据所有线的一般表达式求所有线的交点
    * @param 线的一般表达式
    * @param 输出特征点的ID号
    * @return 输出线交点的像素坐标
    */
    void linesIntersections(const std::vector<Line>& lines,  std::vector<int>& feature_ids, 
                            std::vector<cv::Point2f>& feature_pixels);

    /**
     * 给两条线的表达，求线的交点
     * @param 线的表达1
     * @param 线的表达2
     * @return 线的交点
     */
    cv::Point2f lineIntersection(const Line& line1, const Line& line2);

    /**
     * 根据输入信息，解析出对应的图片参数
     * @param 输入信息
     * @param 图片的尺寸
     * @param 图片的宽度
     * @param 图片的通道数
     * @return 是否解析成功
     */
    bool parseHWFromShape(const std::vector<int>& shape, int& h, int& w, int& c);

public:
    // 点的世界坐标（前右下）
    const cv::Point3f KEYPOINT_WORLD[21] = {
        {0.000f, -5.485f, 0.000f}, // KP0
        {0.000f, -4.115f, 0.000f}, // KP1
        {0.000f, 0.000f, 0.000f}, // KP2
        {0.000f, 4.115f, 0.000f}, // KP3
        {0.000f, 5.485f, 0.000f}, // KP4
        {-6.400f, -4.115f, 0.000f}, // KP5
        {-6.400f, 0.000f, 0.000f}, // KP6
        {-6.400f, 4.115f, 0.000f}, // KP7
        {-11.885f, -5.485f, 0.000f}, // KP8
        {-11.885f, -4.115f, 0.000f}, // KP9
        {-11.885f, 4.115f, 0.000f},  // KP10
        {-11.885f, 5.485f, 0.000f}, // KP11
        {0.000f, -6.399f, 0.000f}, // KP12
        {0.000f,  6.399f, 0.000f}, // KP13
        {0.000f, -6.399f, -1.070f}, // KP14
        {0.000f, -5.485f, -1.029f}, // KP15
        {0.000f, -4.115f, -0.979f}, // KP16
        {0.000f, 0.000f, -0.914f},  // KP17
        {0.000f, 4.115f, -0.979f}, // KP18
        {0.000f, 5.485f, -1.029f}, // KP19
        {0.000f, 6.399f, -1.070f}, // KP20
    };

private:
    // 当前帧的时间戳
    double timestamp_ = 0.0;

    // 网络要求输入图片的尺寸和通道数
    int input_w_ = 0, input_h_ = 0, input_c_ = 0;

    // 输入图片所占字节
    size_t input_bytes_ = 0;

    // 网络的配置参数
    ax_yolo_obb_sync_infer_node::infer_config yolo_config_;

    // 定义网络模型对象
    ax_yolo_obb_sync_infer_node yolo_infer_;

    // 相机指针
    std::vector<std::shared_ptr<GenericCamera>> cameras_;

    // 左右相机的焦距
    std::vector<double> focal_lengths_ = std::vector<double>();

    // 标志位，是否设置了配置文件
    bool restore_cfg_inited_ = false;

    // 关闭时不保留输入图和绘图数据，避免无用 clone/copy。
    bool publish_debug_image_ = false;

    // reloc 输入图像分辨率（与配置文件 reloc.resolution 一致）
    int image_width_ = 0;
    int image_height_ = 0;

    // 定义容器，保存特征点的信息(特征点ID,特征点像素坐标)
    std::vector<int> feature_ids_cam0_ = std::vector<int>(); 
    std::vector<int> feature_ids_cam1_ = std::vector<int>();

    std::vector<cv::Point2f> feature_pts_cam0_ = std::vector<cv::Point2f>();
    std::vector<cv::Point2f> feature_pts_cam1_ = std::vector<cv::Point2f>();

    RelocDebugInfo debug_info_;
    YoloFrameTiming last_frame_timing_;
    std::array<int, 2> last_line_counts_{{0, 0}};
};

}  // namespace ms_fusion

#endif // MS_FUSION_VISION_YOLO_OBB_H
