#include "ms_fusion/common/earth.h"

namespace ms_fusion {

double Earth::gravity(const Eigen::Vector3d& blh) {
    double sin2 = sin(blh[0]);
    sin2 *= sin2;

    return 9.7803267715 * (1 + 0.0052790414 * sin2 + 0.0000232718 * sin2 * sin2)
           + blh[2] * (0.0000000043977311 * sin2 - 0.0000030876910891)
           + 0.0000000000007211 * blh[2] * blh[2];
}

Eigen::Vector2d Earth::meridianPrimeVerticalRadius(double lat) {
    double tmp, sqrttmp;

    tmp = sin(lat);
    tmp *= tmp;
    tmp = 1 - WGS84_E1 * tmp;
    sqrttmp = sqrt(tmp);

    return {WGS84_RA * (1 - WGS84_E1) / (sqrttmp * tmp), WGS84_RA / sqrttmp};
}

double Earth::RN(double lat) {
    double sinlat = sin(lat);
    return WGS84_RA / sqrt(1.0 - WGS84_E1 * sinlat * sinlat);
}

Eigen::Matrix3d Earth::cne(const Eigen::Vector3d& blh) {
    double coslon, sinlon, coslat, sinlat;

    sinlat = sin(blh[0]);
    sinlon = sin(blh[1]);
    coslat = cos(blh[0]);
    coslon = cos(blh[1]);

    Eigen::Matrix3d dcm;
    dcm(0, 0) = -sinlat * coslon;
    dcm(0, 1) = -sinlon;
    dcm(0, 2) = -coslat * coslon;

    dcm(1, 0) = -sinlat * sinlon;
    dcm(1, 1) = coslon;
    dcm(1, 2) = -coslat * sinlon;

    dcm(2, 0) = coslat;
    dcm(2, 1) = 0;
    dcm(2, 2) = -sinlat;

    return dcm;
}

Eigen::Quaterniond Earth::qne(const Eigen::Vector3d& blh) {
    Eigen::Quaterniond quat;

    double coslon, sinlon, coslat, sinlat;

    coslon = cos(blh[1] * 0.5);
    sinlon = sin(blh[1] * 0.5);
    coslat = cos(-M_PI * 0.25 - blh[0] * 0.5);
    sinlat = sin(-M_PI * 0.25 - blh[0] * 0.5);

    quat.w() = coslat * coslon;
    quat.x() = -sinlat * sinlon;
    quat.y() = sinlat * coslon;
    quat.z() = coslat * sinlon;

    return quat.normalized();
}

Eigen::Vector3d Earth::blh(const Eigen::Quaterniond& qne, double height) {
    return {-2 * atan(qne.y() / qne.w()) - M_PI * 0.5, 2 * atan2(qne.z(), qne.w()), height};
}

Eigen::Vector3d Earth::blh2ecef(const Eigen::Vector3d& blh) {
    double coslat, sinlat, coslon, sinlon;
    double rnh, rn;

    coslat = cos(blh[0]);
    sinlat = sin(blh[0]);
    coslon = cos(blh[1]);
    sinlon = sin(blh[1]);

    rn = RN(blh[0]);
    rnh = rn + blh[2];

    return {rnh * coslat * coslon, rnh * coslat * sinlon, (rnh - rn * WGS84_E1) * sinlat};
}

Eigen::Vector3d Earth::ecef2blh(const Eigen::Vector3d& ecef) {
    double p = sqrt(ecef[0] * ecef[0] + ecef[1] * ecef[1]);
    double rn;
    double lat, lon;
    double h = 0, h2;

    // 初始状态
    lat = atan(ecef[2] / (p * (1.0 - WGS84_E1)));
    lon = 2.0 * atan2(ecef[1], ecef[0] + p);

    do {
        h2 = h;
        rn = RN(lat);
        h = p / cos(lat) - rn;
        lat = atan(ecef[2] / (p * (1.0 - WGS84_E1 * rn / (rn + h))));
    } while (fabs(h - h2) > 1.0e-4);

    return {lat, lon, h};
}

Eigen::Matrix3d Earth::DRi(const Eigen::Vector3d& blh) {
    Eigen::Matrix3d dri = Eigen::Matrix3d::Zero();

    Eigen::Vector2d rmn = meridianPrimeVerticalRadius(blh[0]);

    dri(0, 0) = 1.0 / (rmn[0] + blh[2]);
    dri(1, 1) = 1.0 / ((rmn[1] + blh[2]) * cos(blh[0]));
    dri(2, 2) = -1;
    return dri;
}

Eigen::Matrix3d Earth::DR(const Eigen::Vector3d& blh) {
    Eigen::Matrix3d dr = Eigen::Matrix3d::Zero();

    Eigen::Vector2d rmn = meridianPrimeVerticalRadius(blh[0]);

    dr(0, 0) = rmn[0] + blh[2];
    dr(1, 1) = (rmn[1] + blh[2]) * cos(blh[0]);
    dr(2, 2) = -1;
    return dr;
}

Eigen::Vector3d Earth::local2global(const Eigen::Vector3d& origin, const Eigen::Vector3d& local) {
    Eigen::Vector3d ecef0 = blh2ecef(origin);
    Eigen::Matrix3d cn0e = cne(origin);

    Eigen::Vector3d ecef1 = ecef0 + cn0e * local;
    Eigen::Vector3d blh1 = ecef2blh(ecef1);

    return blh1;
}

Eigen::Vector3d Earth::global2local(const Eigen::Vector3d& origin, const Eigen::Vector3d& global) {
    Eigen::Vector3d ecef0 = blh2ecef(origin);
    Eigen::Matrix3d cn0e = cne(origin);

    Eigen::Vector3d ecef1 = blh2ecef(global);

    return cn0e.transpose() * (ecef1 - ecef0);
}

Eigen::Vector3d Earth::iewe() {
    return {0, 0, WGS84_WIE};
}

Eigen::Vector3d Earth::iewn(double lat) {
    return {WGS84_WIE * cos(lat), 0, -WGS84_WIE * sin(lat)};
}

Eigen::Vector3d Earth::iewn(const Eigen::Vector3d& origin, const Eigen::Vector3d& local) {
    Eigen::Vector3d global = local2global(origin, local);

    return iewn(global[0]);
}

Eigen::Vector3d Earth::enwn(const Eigen::Vector2d& rmn, const Eigen::Vector3d& blh,
                            const Eigen::Vector3d& vel) {
    return {vel[1] / (rmn[1] + blh[2]), -vel[0] / (rmn[0] + blh[2]),
            -vel[1] * tan(blh[0]) / (rmn[1] + blh[2])};
}

Eigen::Vector3d Earth::enwn(const Eigen::Vector3d& origin, const Eigen::Vector3d& local,
                            const Eigen::Vector3d& vel) {
    Eigen::Vector3d global = local2global(origin, local);
    Eigen::Vector2d rmn = meridianPrimeVerticalRadius(global[0]);

    return enwn(rmn, global, vel);
}

} // namespace ms_fusion