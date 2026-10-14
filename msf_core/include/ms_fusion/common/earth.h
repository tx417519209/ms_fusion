#ifndef MS_FUSION_EARTH_H
#define MS_FUSION_EARTH_H

#include "eigen_utils.hpp"

namespace ms_fusion {

class Earth
{
public:
    static double gravity(const Eigen::Vector3d& blh);

    static Eigen::Vector2d meridianPrimeVerticalRadius(double lat);
    static double RN(double lat);

    static Eigen::Matrix3d cne(const Eigen::Vector3d& blh);
    static Eigen::Quaterniond qne(const Eigen::Vector3d& blh);
    static Eigen::Vector3d blh(const Eigen::Quaterniond& qne, double height);

    static Eigen::Vector3d blh2ecef(const Eigen::Vector3d& blh);
    static Eigen::Vector3d ecef2blh(const Eigen::Vector3d& ecef);

    static Eigen::Matrix3d DRi(const Eigen::Vector3d& blh);
    static Eigen::Matrix3d DR(const Eigen::Vector3d& blh);

    static Eigen::Vector3d local2global(const Eigen::Vector3d& origin,
                                        const Eigen::Vector3d& local);
    static Eigen::Vector3d global2local(const Eigen::Vector3d& origin,
                                        const Eigen::Vector3d& global);

    static Eigen::Vector3d iewe();
    static Eigen::Vector3d iewn(double lat);
    static Eigen::Vector3d iewn(const Eigen::Vector3d& origin, const Eigen::Vector3d& local);
    static Eigen::Vector3d enwn(const Eigen::Vector2d& rmn, const Eigen::Vector3d& blh,
                                const Eigen::Vector3d& vel);
    static Eigen::Vector3d enwn(const Eigen::Vector3d& origin, const Eigen::Vector3d& local,
                                const Eigen::Vector3d& vel);

private:
    static constexpr double WGS84_WIE = 7.2921151467E-5;      // 地球自转角速度
    static constexpr double WGS84_F = 0.0033528106647474805;  // 扁率
    static constexpr double WGS84_RA = 6378137.0000000000;    // 长半轴a
    static constexpr double WGS84_RB = 6356752.3142451793;    // 短半轴b
    static constexpr double WGS84_GM0 = 398600441800000.00;   // 地球引力常数
    static constexpr double WGS84_E1 = 0.0066943799901413156; // 第一偏心率平方
    static constexpr double WGS84_E2 = 0.0067394967422764341; // 第二偏心率平方
};

} // namespace ms_fusion

#endif // MS_FUSION_EARTH_H
