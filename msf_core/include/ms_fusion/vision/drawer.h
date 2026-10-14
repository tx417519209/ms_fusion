#ifndef MS_FUSION_VISION_DRAWER_H
#define MS_FUSION_VISION_DRAWER_H

#include "ms_fusion/common/measurement.h"

#include <opencv2/opencv.hpp>

namespace ms_fusion {

class Drawer {
public:
    static cv::Mat drawFeatureDetect(const cv::Mat& image, const std::vector<cv::Point2f>& pts);

    // 在彩色或灰度图上绘制关键点（NPU 原始点 + 可选网格筛选点）
    static cv::Mat drawKeypointsOnImage(const cv::Mat& image,
                                        const std::vector<cv::Point2f>& npu_pts,
                                        const std::vector<cv::Point2f>& grid_picked_pts = {});

    static cv::Mat drawFeatureStereo(const cv::Mat& image0, const cv::Mat& image1, 
                                     const std::vector<cv::Point2f>& pts0, 
                                     const std::vector<cv::Point2f>& pts1);

    // 双目特征跟踪可视化（匹配线 + ID）
    static cv::Mat drawTrackingStereoDebug(const cv::Mat& image0,
                                           const cv::Mat& image1,
                                           const std::vector<cv::Point2f>& pts0,
                                           const std::vector<cv::Point2f>& pts1,
                                           const std::vector<size_t>& ids,
                                           const std::vector<cv::Point2f>& detect_keypoints = {});

    // 左目 t-1→t 时间跟踪可视化：单张当前帧画布，每条轨迹画 prev->curr 箭头（风格参考
    // front_end ROSViewer::drawerTemporalTrack，不再用 t-1|t 并排画布）。matched（绿）=
    // 延续轨迹，new（黄，prev==curr，零长箭头即一个点）=本轮新补充且首次匹配成功，
    // failed_prev（红叉，画在其最后已知位置）=本轮匹配失败/被剔除、已经没有当前帧位置的轨迹
    static cv::Mat drawTemporalTracking(const cv::Mat& image_curr,
                                        const std::vector<cv::Point2f>& matched_prev,
                                        const std::vector<cv::Point2f>& matched_curr,
                                        const std::vector<cv::Point2f>& new_prev,
                                        const std::vector<cv::Point2f>& new_curr,
                                        const std::vector<cv::Point2f>& failed_prev);

    // 双目 MNN 匹配调试图：左(image0)|右(image1) 横向拼接，pts0[i]<->pts1[i] 逐对连线，
    // 按 ids[i]（左图 track ID）上色，同一物理点跨帧颜色保持一致
    static cv::Mat drawStereoMatchDebug(const cv::Mat& image0, const cv::Mat& image1,
                                        const std::vector<cv::Point2f>& pts0,
                                        const std::vector<cv::Point2f>& pts1,
                                        const std::vector<size_t>& ids);

    // YOLO-OBB 线段 + 重定位特征点可视化
    static cv::Mat drawRelocStereoDebug(const cv::Mat& image0,
                                        const cv::Mat& image1,
                                        const RelocDebugInfo& reloc);

    // 拼接 tracking 与 reloc 可视化面板
    static cv::Mat composeVisionDebugImage(const TrackingDebugInfo& tracking,
                                           const RelocDebugInfo* reloc,
                                           bool draw_reloc);
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_DRAWER_H