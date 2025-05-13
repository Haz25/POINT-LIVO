/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <Eigen/Eigen>
#include "common_lib.h"
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <utils/so3_math.h>
#include "voxel_map.h"
#include <pcl/filters/voxel_grid.h>

class VoxelMapManager;
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

//#include "LIVMapper.h"

const bool time_list(PointType &x, PointType &y); //{return (x.curvature < y.curvature);};
std::vector<int> time_compressing(const PointCloudXYZI::Ptr &point_cloud);                     

enum Type {IMU, LIDAR, DEFAULT};

struct measure {
  int idx;
  double time;
  Type type;

  bool operator<(measure &b) {
    return time < b.time;
  }
};

/// *************IMU Process and undistortion
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov_scale(const V3D &scaler);
  void set_acc_cov_scale(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_inv_expo_cov(const double &inv_expo);
  void set_imu_init_frame_num(const int &num);
  void set_cov(const double inv_expo_cov,
               const double vel_cov,
               const double gyr_state_cov, const double acc_state_cov, 
               const double bias_gyr_cov, const double bias_acc_cov);
  void set_R();
  void set_satu(const double satu_gyr, const double satu_acc);
  void disable_imu();
  void disable_gravity_est();
  void disable_bias_est();
  void disable_exposure_est();
  void Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_, 
                vector<pointWithVar> &_pv_list, VoxelMapManagerPtr &voxelmap_manager);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  void UndistortPclPointLIO(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);  
  void UndistortPclCustom(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out, VoxelMapManagerPtr& voxelmap_manager);               
  //void processImu(StatesGroup &stat);
  void Predict(StatesGroup &stat, double dt, bool predict_state, bool prop_cov);
  void StateEstimationIMU(StatesGroup &stat);

  ofstream fout_imu;
  double IMU_mean_acc_norm;
  V3D unbiased_gyr;

  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double vel_cov = 10.0;
  double gyr_cov = 0.3;
  double acc_cov = 0.3;
  double gyr_state_cov = 500.0;
  double acc_state_cov = 1000.0;
  double bias_gyr_cov = 0.0001;
  double bias_acc_cov = 0.0001;
  double cov_inv_expo;
  double satu_gyr = 50.0;
  double satu_acc = 50.0;
  double first_lidar_time;
  bool imu_time_init = false;
  bool imu_need_init = true;
  M3D Eye3d;
  V3D Zero3d;
  int lidar_type;
  MD(DIM_STATE, DIM_STATE) cov_w;
  MD(6, 6) R_IMU;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  SLAM_MODE slam_mode;
  bool lidar_map_inited = false;

private:
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  PointCloudXYZI pcl_wait_proc;
  sensor_msgs::ImuConstPtr last_imu;
  PointCloudXYZI::Ptr cur_pcl_un_;
  PointCloudXYZI::Ptr pcl_filtered;
  vector<Pose6D> IMUpose;
  M3D Lid_rot_to_IMU;
  V3D Lid_offset_to_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  V3D omg_meas;
  V3D acc_meas;
  V3D alpha = V3D::Zero();
  V3D jerk = V3D::Zero();
  double last_prop_end_time;
  double time_last_scan;
  int init_iter_num = 1, MAX_INI_COUNT = 20;
  bool b_first_frame = true;
  bool imu_en = true;
  bool gravity_est_en = true;
  bool ba_bg_est_en = true;
  bool exposure_estimate_en = true;
  double last_prop_time;
  double last_update_time;
};
typedef std::shared_ptr<ImuProcess> ImuProcessPtr;
#endif