/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "IMU_Processing.h"

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); }

std::vector<int> time_compressing(const PointCloudXYZI::Ptr &point_cloud) {
  int points_size = point_cloud->points.size();
  int j = 0;
  std::vector<int> time_seq;
  time_seq.reserve(points_size);
  for(int i = 0; i < points_size - 1; i++) {
    j++;
    if (point_cloud->points[i+1].curvature > point_cloud->points[i].curvature) {
      time_seq.emplace_back(j);
      j = 0;
    }
  }
  time_seq.emplace_back(j+1);
  return time_seq;
}

ImuProcess::ImuProcess() : Eye3d(M3D::Identity()),
                           Zero3d(0, 0, 0), b_first_frame(true), imu_need_init(true)
{
  init_iter_num = 1;
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_acc = V3D(0.1, 0.1, 0.1);
  cov_inv_expo = 0.2;
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  acc_s_last = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU = Eye3d;
  last_imu.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
  pcl_filtered.reset(new PointCloudXYZI());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
  ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init = true;
  init_iter_num = 1;
  IMUpose.clear();
  last_imu.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
  pcl_filtered.reset(new PointCloudXYZI());
}

void ImuProcess::disable_imu()
{
  cout << "IMU Disabled !!!!!" << endl;
  imu_en = false;
  imu_need_init = false;
}

void ImuProcess::disable_gravity_est()
{
  cout << "Online Gravity Estimation Disabled !!!!!" << endl;
  gravity_est_en = false;
}

void ImuProcess::disable_bias_est()
{
  cout << "Bias Estimation Disabled !!!!!" << endl;
  ba_bg_est_en = false;
}

void ImuProcess::disable_exposure_est()
{
  cout << "Online Time Offset Estimation Disabled !!!!!" << endl;
  exposure_estimate_en = false;
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T)
{
  Lid_offset_to_IMU = T.block<3, 1>(0, 3);
  Lid_rot_to_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) { cov_gyr = scaler; }

void ImuProcess::set_acc_cov_scale(const V3D &scaler) { cov_acc = scaler; }

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_inv_expo_cov(const double &inv_expo) { cov_inv_expo = inv_expo; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

void ImuProcess::set_imu_init_frame_num(const int &num) { MAX_INI_COUNT = num; }

void ImuProcess::set_cov(const double inv_expo_cov,
                         const double vel_cov, 
                         const double gyr_state_cov, const double acc_state_cov,
                         const double bias_gyr_cov, const double bias_acc_cov) {
  
  cov_w.setZero();
  cov_w(6, 6) = inv_expo_cov;
  cov_w.block<3, 3>(10, 10) = bias_gyr_cov * M3D::Identity();
  cov_w.block<3, 3>(13, 13) = bias_acc_cov * M3D::Identity(); 
  
  //cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr;
  //cov_w.block<3, 3>(7, 7).diagonal() = cov_acc;
  
  cov_w.block<3, 3>(7, 7) = vel_cov * M3D::Identity();
  cov_w.block<3, 3>(19, 19) = gyr_state_cov * M3D::Identity();
  cov_w.block<3, 3>(22, 22) = acc_state_cov * M3D::Identity();
};

void ImuProcess::set_R() {
  R_IMU.setZero();
  R_IMU.block<3, 3>(0, 0).diagonal() = cov_gyr;
  R_IMU.block<3, 3>(3, 3).diagonal() = cov_acc;
};

void ImuProcess::set_satu(const double satu_gyr, const double satu_acc) {
  this->satu_gyr = satu_gyr;
  this->satu_acc = satu_acc;
};

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame)
  {
    Reset();
    N = 1;
    b_first_frame = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    last_prop_time = meas.imu.front()->header.stamp.toSec();
    last_update_time = meas.imu.front()->header.stamp.toSec();
    last_prop_end_time = meas.imu.front()->header.stamp.toSec();
    // first_lidar_time = meas.lidar_frame_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc -
    // mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N); cov_gyr
    // = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr -
    // mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }
  IMU_mean_acc_norm = mean_acc.norm();
  state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;
  state_inout.acc = -state_inout.gravity;
  state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g = Zero3d; // mean_gyr;

  last_imu = meas.imu.back();
}

void ImuProcess::Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  pcl_out = *(meas.lidar);
  /*** sort point clouds by offset time ***/
  const double &pcl_beg_time = meas.lidar_frame_beg_time;
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  meas.last_lio_update_time = pcl_end_time;
  const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);

  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt = 0;

  if (b_first_frame)
  {
    dt = 0.1;
    b_first_frame = false;
  }
  else { dt = pcl_beg_time - time_last_scan; }

  time_last_scan = pcl_beg_time;
  // for (size_t i = 0; i < pcl_out->points.size(); i++) {
  //   if (dt < pcl_out->points[i].curvature) {
  //     dt = pcl_out->points[i].curvature;
  //   }
  // }
  // dt = dt / (double)1000;
  // std::cout << "dt:" << dt << std::endl;
  // double dt = pcl_out->points.back().curvature / double(1000);

  /* covariance propagation */
  // M3D acc_avr_skew;
  M3D Exp_f = Exp(state_inout.bias_g, dt);

  F_x.setIdentity();
  cov_w.setZero();

  F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
  F_x.block<3, 3>(0, 10) = Eye3d * dt;
  F_x.block<3, 3>(3, 7) = Eye3d * dt;
  // F_x.block<3, 3>(6, 0)  = - R_imu * acc_avr_skew * dt;
  // F_x.block<3, 3>(6, 12) = - R_imu * dt;
  // F_x.block<3, 3>(6, 15) = Eye3d * dt;

  cov_w.block<3, 3>(10, 10).diagonal() = cov_gyr * dt * dt; // for omega in constant model
  cov_w.block<3, 3>(7, 7).diagonal() = cov_acc * dt * dt; // for velocity in constant model
  // cov_w.block<3, 3>(6, 6) =
  //     R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
  // cov_w.block<3, 3>(9, 9).diagonal() =
  //     cov_bias_gyr * dt * dt; // bias gyro covariance
  // cov_w.block<3, 3>(12, 12).diagonal() =
  //     cov_bias_acc * dt * dt; // bias acc covariance

  // std::cout << "before propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
  // std::cout << "cov_w:" << cov_w.diagonal().transpose() << std::endl;
  // std::cout << "after propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.rot = state_inout.rot * Exp_f;
  state_inout.pos = state_inout.pos + state_inout.vel * dt;

  if (lidar_type != L515)
  {
    auto it_pcl = pcl_out.points.end() - 1;
    double dt_j = 0.0;
    for(; it_pcl != pcl_out.points.begin(); it_pcl--)
    {
        dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
        M3D R_jk(Exp(state_inout.bias_g, - dt_j));
        V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
        // Using rotation and translation to un-distort points
        V3D p_jk;
        p_jk = - state_inout.rot.transpose() * state_inout.vel * dt_j;
  
        V3D P_compensate =  R_jk * P_j + p_jk;
  
        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
    }
  }
}

void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  // printf("[ IMU ] undistort input size: %zu \n", lidar_meas.pcl_proc_cur->points.size());
  // printf("[ IMU ] IMU data sequence size: %zu \n", meas.imu.size());
  // printf("[ IMU ] lidar_scan_index_now: %d \n", lidar_meas.lidar_scan_index_now);

  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  /*** cut lidar point based on the propagation-start time and required
   * propagation-end time ***/
  // const double pcl_offset_time = (prop_end_time -
  // lidar_meas.lidar_frame_beg_time) * 1000.; // the offset time w.r.t scan
  // start time auto pcl_it = lidar_meas.pcl_proc_cur->points.begin() +
  // lidar_meas.lidar_scan_index_now; auto pcl_it_end =
  // lidar_meas.lidar->points.end(); printf("[ IMU ] pcl_it->curvature: %lf
  // pcl_offset_time: %lf \n", pcl_it->curvature, pcl_offset_time); while
  // (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  // {
  //   pcl_wait_proc.push_back(*pcl_it);
  //   pcl_it++;
  //   lidar_meas.lidar_scan_index_now++;
  // }

  // cout<<"pcl_out.size(): "<<pcl_out.size()<<endl;
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:
  // "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;

  // printf("[ IMU ] last propagation end time: %lf \n", lidar_meas.last_lio_update_time);
  if (lidar_meas.lio_vio_flg == LIO)
  {
    pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel, state_inout.pos, state_inout.rot));
  }

  // printf("[ IMU ] pcl_wait_proc size: %zu \n", pcl_wait_proc.points.size());

  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU ]: Process lidar from "<<prop_beg_time<<" to
  // "<<prop_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to
  //          "<<imu_end_time<<endl;
  // cout<<"[ IMU ]: point size: "<<lidar_meas.lidar->points.size()<<endl;

  /*** Initialize IMU pose ***/
  // IMUpose.clear();

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  // cout << "[ IMU ] input state: " << state_inout.vel.transpose() << " " << state_inout.pos.transpose() << endl;
  M3D R_imu(state_inout.rot);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;
  // double imu_time;
  double tau;
  if (!imu_time_init)
  {
    // imu_time = v_imu.front()->header.stamp.toSec() - first_lidar_time;
    // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
    tau = 1.0;
    imu_time_init = true;
  }
  else
  {
    tau = state_inout.inv_expo_time;
    // ROS_ERROR("tau: %.6f !!!!!!", tau);
  }
  // state_inout.cov(6, 6) = 0.01;

  // ROS_ERROR("lidar_meas.lio_vio_flg");
  // cout<<"lidar_meas.lio_vio_flg: "<<lidar_meas.lio_vio_flg<<endl;
  switch (lidar_meas.lio_vio_flg)
  {
  case LIO:
  case VIO:
    dt = 0;
    for (int i = 0; i < v_imu.size() - 1; i++)
    {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      if (tail->header.stamp.toSec() < prop_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

      //angvel_avr << tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z;

      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      //acc_avr << tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z;

      // cout<<"angvel_avr: "<<angvel_avr.transpose()<<endl;
      // cout<<"acc_avr: "<<acc_avr.transpose()<<endl;

      // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
      // #endif

      // imu_time = head->header.stamp.toSec() - first_lidar_time;

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      if (head->header.stamp.toSec() < prop_beg_time)
      {
        // printf("00 \n");
        dt = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2)
      {
        // printf("11 \n");
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else
      {
        // printf("22 \n");
        dt = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }

      dt_all += dt;
      // printf("[ LIO Propagation ] dt: %lf \n", dt);

      /* covariance propagation */
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
      if (ba_bg_est_en) F_x.block<3, 3>(0, 10) = -Eye3d * dt;
      // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
      F_x.block<3, 3>(3, 7) = Eye3d * dt;
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;
      if (ba_bg_est_en) F_x.block<3, 3>(7, 13) = -R_imu * dt;
      if (gravity_est_en) F_x.block<3, 3>(7, 16) = Eye3d * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
      // F_x(6,6) = 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) * (-tau*tau); F_x(18,18) = 0.00001;
      if (exposure_estimate_en) cov_w(6, 6) = cov_inv_expo * dt * dt;
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
      // state_inout.cov.block<18,18>(0,0) = F_x.block<18,18>(0,0) *
      // state_inout.cov.block<18,18>(0,0) * F_x.block<18,18>(0,0).transpose() +
      // cov_w.block<18,18>(0,0);

      // tau = tau + 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) *
      // (-tau*tau) * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);

      /* propogation of IMU attitude */
      R_imu = R_imu * Exp_f;

      /* Specific acceleration (global frame) of IMU */
      acc_imu = R_imu * acc_avr + state_inout.gravity;

      /* propogation of IMU */
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /* velocity of IMU */
      vel_imu = vel_imu + acc_imu * dt;

      /* save the poses at each IMU measurements */
      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
      // "<<tail->header.stamp.toSec()<<endl; printf("[ LIO Propagation ]
      // offs_t: %lf \n", offs_t);
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    // unbiased_gyr = V3D(IMUpose.back().gyr[0], IMUpose.back().gyr[1], IMUpose.back().gyr[2]);
    // cout<<"prop end - start: "<<prop_end_time - prop_beg_time<<" dt_all: "<<dt_all<<endl;
    lidar_meas.last_lio_update_time = prop_end_time;
    // dt = prop_end_time - imu_end_time;
    // printf("[ LIO Propagation ] dt: %lf \n", dt);
    break;
  }

  state_inout.vel = vel_imu;
  state_inout.rot = R_imu;
  state_inout.pos = pos_imu;
  state_inout.inv_expo_time = tau;

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // if (imu_end_time>prop_beg_time)
  // {
  //   double note = prop_end_time > imu_end_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - imu_end_time);
  //   state_inout.vel = vel_imu + note * acc_imu * dt;
  //   state_inout.rot = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }
  // else
  // {
  //   double note = prop_end_time > prop_beg_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - prop_beg_time);
  //   state_inout.vel = vel_imu + note * acc_imu * dt;
  //   state_inout.rot = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }

  // cout<<"[ Propagation ] output state: "<<state_inout.vel.transpose() <<
  // state_inout.pos.transpose()<<endl;

  last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  // auto pos_liD_e = state_inout.pos + state_inout.rot *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot * Lidar_R_to_IMU;

  // cout<<"[ IMU ]: vel "<<state_inout.vel.transpose()<<" pos
  // "<<state_inout.pos.transpose()<<"
  // ba"<<state_inout.bias_a.transpose()<<" bg
  // "<<state_inout.bias_g.transpose()<<endl; cout<<"propagated cov:
  // "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO
   * update ***/
  if (lidar_meas.lio_vio_flg == LIO)
  {
    auto it_pcl = pcl_wait_proc.points.end() - 1;
    M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      // printf("head->offset_time: %lf \n", head->offset_time);
      // printf("it_pcl->curvature: %lf pt dt: %lf \n", it_pcl->curvature,
      // it_pcl->curvature / double(1000) - head->offset_time);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame */
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos);

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        // V3D P_compensate = Lid_rot_to_IMU.transpose() *
        // (state_inout.rot.transpose() * (R_i * (Lid_rot_to_IMU * P_i +
        // Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);
        V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_wait_proc.points.begin()) break;
      }
    }
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
  // printf("[ IMU ] time forward: %lf, backward: %lf.\n", t1 - t0, omp_get_wtime() - t1);
}

void ImuProcess::UndistortPclPointLIO(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  // printf("[ IMU ] undistort input size: %zu \n", lidar_meas.pcl_proc_cur->points.size());
  // printf("[ IMU ] IMU data sequence size: %zu \n", meas.imu.size());
  // printf("[ IMU ] lidar_scan_index_now: %d \n", lidar_meas.lidar_scan_index_now);

  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  /*** cut lidar point based on the propagation-start time and required
   * propagation-end time ***/
  // const double pcl_offset_time = (prop_end_time -
  // lidar_meas.lidar_frame_beg_time) * 1000.; // the offset time w.r.t scan
  // start time auto pcl_it = lidar_meas.pcl_proc_cur->points.begin() +
  // lidar_meas.lidar_scan_index_now; auto pcl_it_end =
  // lidar_meas.lidar->points.end(); printf("[ IMU ] pcl_it->curvature: %lf
  // pcl_offset_time: %lf \n", pcl_it->curvature, pcl_offset_time); while
  // (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  // {
  //   pcl_wait_proc.push_back(*pcl_it);
  //   pcl_it++;
  //   lidar_meas.lidar_scan_index_now++;
  // }

  // cout<<"pcl_out.size(): "<<pcl_out.size()<<endl;
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:
  // "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;

  // printf("[ IMU ] last propagation end time: %lf \n", lidar_meas.last_lio_update_time);
  if (lidar_meas.lio_vio_flg == LIO)
  {
    pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel, state_inout.pos, state_inout.rot));
  }

  // printf("[ IMU ] pcl_wait_proc size: %zu \n", pcl_wait_proc.points.size());

  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU ]: Process lidar from "<<prop_beg_time<<" to
  // "<<prop_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to
  //          "<<imu_end_time<<endl;
  // cout<<"[ IMU ]: point size: "<<lidar_meas.lidar->points.size()<<endl;

  /*** Initialize IMU pose ***/
  // IMUpose.clear();

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  // cout << "[ IMU ] input state: " << state_inout.vel.transpose() << " " << state_inout.pos.transpose() << endl;
  M3D R_imu(state_inout.rot);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;
  // double imu_time;
  double tau;
  if (!imu_time_init)
  {
    // imu_time = v_imu.front()->header.stamp.toSec() - first_lidar_time;
    // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
    tau = 1.0;
    imu_time_init = true;
  }
  else
  {
    tau = state_inout.inv_expo_time;
    // ROS_ERROR("tau: %.6f !!!!!!", tau);
  }
  // state_inout.cov(6, 6) = 0.01;

  // ROS_ERROR("lidar_meas.lio_vio_flg");
  // cout<<"lidar_meas.lio_vio_flg: "<<lidar_meas.lio_vio_flg<<endl;

  cout << "before propagate" << endl;
  cout << "R_imu:\n" << R_imu << endl;
  cout << "pos_imu: " << pos_imu.transpose() << endl;

  switch (lidar_meas.lio_vio_flg) {
  case LIO:
  
    lidar_meas.last_lio_update_time = prop_end_time;
    last_imu = v_imu.back();
    last_prop_end_time = prop_end_time;
    if (pcl_wait_proc.points.size() < 1) return;
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
    break;
    
  // =========================== need to be changed ========================== // 
  case VIO:  
    dt = 0;
    cout << " we are in UndistortPCL " << endl;
    cout << "prop_beg_time: " << prop_beg_time << endl;
    acc_imu = R_imu * state_inout.acc + state_inout.gravity;
    for (int i = 0; i < v_imu.size() - 1; i++) {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      auto &tmp = head;
      cout << i << "-th imu_data:\n linear: " << tmp->linear_acceleration.x << " "
                                           << tmp->linear_acceleration.y << " "
                                           << tmp->linear_acceleration.z << " "
                           << " angular: " << tmp->angular_velocity.x << " "
                                           << tmp->angular_velocity.y << " "
                                           << tmp->angular_velocity.z << " "
                              << " time: " << tmp->header.stamp.toNSec() << endl;
      

      if (tail->header.stamp.toSec() < prop_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
      //angvel_avr << head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z;

      // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y,
      // tail->angular_velocity.z;

      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
      //acc_avr << head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z;

      // cout<<"angvel_avr: "<<angvel_avr.transpose()<<endl;
      // cout<<"acc_avr: "<<acc_avr.transpose()<<endl;

      // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
      // #endif

      // imu_time = head->header.stamp.toSec() - first_lidar_time;

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      if (head->header.stamp.toSec() < prop_beg_time) {
        dt = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2) {
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else {
        dt = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }

      dt_all += dt;
      // printf("[ LIO Propagation ] dt: %lf \n", dt);

      /* covariance propagation */
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();

      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);            // (rot, rot)
      if (ba_bg_est_en) F_x.block<3, 3>(0, 10) = -Eye3d * dt;  // (rot, bias_g)
      // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
      F_x.block<3, 3>(3, 7) = Eye3d * dt;                      // (pos, vel)
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;      // (vel, rot)
      if (ba_bg_est_en) F_x.block<3, 3>(7, 13) = -R_imu * dt;  // (vel, bias_a)
      if (gravity_est_en) F_x.block<3, 3>(7, 16) = Eye3d * dt; // (vel, gravity)

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
      // F_x(6,6) = 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) * (-tau*tau); F_x(18,18) = 0.00001;
      if (exposure_estimate_en) cov_w(6, 6) = cov_inv_expo * dt * dt;                       // inv_expo_time covariance
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;                               // gyro covariance
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt; // acc covariance
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt * dt;                        // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt * dt;                        // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
      // state_inout.cov.block<18,18>(0,0) = F_x.block<18,18>(0,0) *
      // state_inout.cov.block<18,18>(0,0) * F_x.block<18,18>(0,0).transpose() +
      // cov_w.block<18,18>(0,0);

      // tau = tau + 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) *
      // (-tau*tau) * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);

      /* propogation of IMU attitude */
      R_imu = R_imu * Exp_f;

      /* Specific acceleration (global frame) of IMU */
      acc_imu = R_imu * acc_avr + state_inout.gravity;

      /* propogation of IMU */
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /* velocity of IMU */
      vel_imu = vel_imu + acc_imu * dt;

      /* save the poses at each IMU measurements */
      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
      // "<<tail->header.stamp.toSec()<<endl; printf("[ LIO Propagation ]
      // offs_t: %lf \n", offs_t);
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    // unbiased_gyr = V3D(IMUpose.back().gyr[0], IMUpose.back().gyr[1], IMUpose.back().gyr[2]);
    // cout<<"prop end - start: "<<prop_end_time - prop_beg_time<<" dt_all: "<<dt_all<<endl;
    lidar_meas.last_lio_update_time = prop_end_time;
    // dt = prop_end_time - imu_end_time;
    // printf("[ LIO Propagation ] dt: %lf \n", dt);
    break;
  }

  //if (lidar_meas.lio_vio_flg == VIO) {
  state_inout.vel = vel_imu;
  state_inout.rot = R_imu;
  state_inout.pos = pos_imu;
  state_inout.inv_expo_time = tau;
  //}

  cout << "after propagate" << endl;
  cout << "R_imu:\n" << R_imu << endl;
  cout << "pos_imu: " << pos_imu.transpose() << endl;
  

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // if (imu_end_time>prop_beg_time)
  // {
  //   double note = prop_end_time > imu_end_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - imu_end_time);
  //   state_inout.vel = vel_imu + note * acc_imu * dt;
  //   state_inout.rot = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }
  // else
  // {
  //   double note = prop_end_time > prop_beg_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - prop_beg_time);
  //   state_inout.vel = vel_imu + note * acc_imu * dt;
  //   state_inout.rot = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }

  // cout<<"[ Propagation ] output state: "<<state_inout.vel.transpose() <<
  // state_inout.pos.transpose()<<endl;

  last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  // auto pos_liD_e = state_inout.pos + state_inout.rot *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot * Lidar_R_to_IMU;

  // cout<<"[ IMU ]: vel "<<state_inout.vel.transpose()<<" pos
  // "<<state_inout.pos.transpose()<<"
  // ba"<<state_inout.bias_a.transpose()<<" bg
  // "<<state_inout.bias_g.transpose()<<endl; cout<<"propagated cov:
  // "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO
   * update ***/
  // ======================= need to be changed ======================= //
  
  if (lidar_meas.lio_vio_flg == LIO)  
  {
    auto it_pcl = pcl_wait_proc.points.end() - 1;
    M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      // printf("head->offset_time: %lf \n", head->offset_time);
      // printf("it_pcl->curvature: %lf pt dt: %lf \n", it_pcl->curvature,
      // it_pcl->curvature / double(1000) - head->offset_time);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        // Transform to the 'end' frame //
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos);

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        // V3D P_compensate = Lid_rot_to_IMU.transpose() *
        // (state_inout.rot.transpose() * (R_i * (Lid_rot_to_IMU * P_i +
        // Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);
        V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_wait_proc.points.begin()) break;

        // what if stateEstimation is here?
      }
    }
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
  
  // printf("[ IMU ] time forward: %lf, backward: %lf.\n", t1 - t0, omp_get_wtime() - t1);
}

void ImuProcess::UndistortPcl2(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out) {
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;

  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  if (lidar_meas.lio_vio_flg == LIO) {
    pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel, state_inout.pos, state_inout.rot));
  }

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  M3D R_imu(state_inout.rot);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;
  double tau;
  if (!imu_time_init) {
    tau = 1.0;
    imu_time_init = true;
  }
  else {
    tau = state_inout.inv_expo_time;
  }

  switch (lidar_meas.lio_vio_flg) {
  case LIO:
  case VIO:
    dt = 0;
    for (int i = 0; i < v_imu.size() - 1; i++) {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      if (tail->header.stamp.toSec() < prop_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      omg_meas = angvel_avr;
      acc_meas = acc_avr * G_m_s2 / mean_acc.norm();

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      if (head->header.stamp.toSec() < prop_beg_time) {
        dt = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2) {
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else {
        dt = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }
      dt_all += dt;

      //cout << "dt: " << dt << endl;
      /* covariance propagation */
      /*
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
      F_x.block<3, 3>(0, 10) = -Eye3d * dt;
      F_x.block<3, 3>(3, 7) = Eye3d * dt;
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;
      F_x.block<3, 3>(7, 13) = -R_imu * dt;
      F_x.block<3, 3>(7, 16) = Eye3d * dt;

      cov_w(6, 6) = cov_inv_expo * dt * dt;
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

      R_imu = R_imu * Exp_f;
      acc_imu = R_imu * acc_avr + state_inout.gravity;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
      */
      
      
      Predict(state_inout, dt, true, true);
      StateEstimationIMU(state_inout);
      R_imu = state_inout.rot;
      acc_imu = state_inout.rot * state_inout.acc + state_inout.gravity;
      //acc_imu = state_inout.rot * acc_avr + state_inout.gravity;
      pos_imu = state_inout.pos;
      vel_imu = state_inout.vel;
      
      
      /* save the poses at each IMU measurements */

      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    lidar_meas.last_lio_update_time = prop_end_time;
    break;
  }

  state_inout.vel = vel_imu;
  state_inout.rot = R_imu;
  state_inout.pos = pos_imu;
  state_inout.inv_expo_time = tau;

  last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO
   * update ***/
  int i = 0;
  if (lidar_meas.lio_vio_flg == LIO) {
    sort(pcl_wait_proc.points.begin(), pcl_wait_proc.points.end(), time_list);
    auto it_pcl = pcl_wait_proc.points.begin();
    M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto it_kp = IMUpose.begin(); it_kp != IMUpose.end() - 1; it_kp++) {
      auto head = it_kp;
      auto tail = it_kp + 1;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      //cout << i << "-th time_offset:" << head->offset_time << endl;

      int cnt = 0;
      for (; it_pcl->curvature / double(1000) < tail->offset_time; it_pcl++) {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame */
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_i(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt);
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        V3D P_compensate = (extR_end * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i) + extT_end);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        cnt ++;
        if (it_pcl == pcl_wait_proc.points.end() - 1) break;
      }
      //cout << "number of points in imu frame: " << cnt << endl;
      i ++;
      //if (i == IMUpose.size() - 1) cout << i << "-th time_offset:" << tail->offset_time << endl;
    }
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
}

void ImuProcess::UndistortPcl3(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out, 
                               vector<pointWithVar> &_pv_list, VoxelMapManagerPtr &voxelmap_manager) {
  double t0 = omp_get_wtime();
  pcl_out.clear();
  _pv_list.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  if (lidar_meas.lio_vio_flg == LIO) {
    //pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    //pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    downSizeFilterSurf.setInputCloud(lidar_meas.pcl_proc_cur);
    downSizeFilterSurf.filter(pcl_wait_proc);
    lidar_meas.lidar_scan_index_now = 0;
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel, state_inout.pos, state_inout.rot));
  }
  sort(pcl_wait_proc.begin(), pcl_wait_proc.end(), time_list);

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  M3D R_imu(state_inout.rot);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_cov, dt_all = 0.0;
  double offs_t;
  double tau;
  PointVector::iterator it_pcl;
  PointCloudXYZI::Ptr pcl_body(new PointCloudXYZI());
  PointCloudXYZI::Ptr pcl_world(new PointCloudXYZI());
  //PointCloudXYZI pcl_wait_proc = pcl_wait_proc;
  //PointCloudXYZI pcl_wait_proc = pcl_wait_proc;
  //std::vector<V3D> pcl_wait_proc_point;
  //std::vector<V3D> pcl_wait_proc_point;

  int max_iterations = voxelmap_manager->config_setting_.max_iterations_;

  if (!imu_time_init) {
    tau = 1.0;
    imu_time_init = true;
  }
  else {
    tau = state_inout.inv_expo_time;
  }

  switch (lidar_meas.lio_vio_flg) {
  case LIO:
  case VIO:
    if (lidar_meas.lio_vio_flg == LIO) {
      it_pcl = pcl_wait_proc.points.begin();
    }
    dt_cov = 0;
    double time_offset = 0.0;
    acc_imu = R_imu * state_inout.acc + state_inout.gravity;
    for (int i = 0; i < v_imu.size() - 1; i++) {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      if (tail->header.stamp.toSec() < prop_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
      //angvel_avr << head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z;
      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
      //acc_avr << head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z;

      omg_meas = angvel_avr;
      acc_meas = acc_avr * G_m_s2 / mean_acc.norm();

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      //cout << "last_prop_end_time: " << last_prop_end_time << endl;
      /* time difference calculation */
      if (head->header.stamp.toSec() < prop_beg_time) {
        dt_cov = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2) {
        dt_cov = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else {
        dt_cov = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }
      dt_all += dt_cov;

      /*
      cout << i << "-th time_offset:" << time_offset << endl;
      cout << "state_inout.omg: " << state_inout.omg.transpose() << endl;
      cout << "state_inout.acc: " << state_inout.acc.transpose() << endl;
      cout << "state_inout.gravity: " << state_inout.gravity.transpose() << endl;
      cout << "angvel_avr: " << angvel_avr.transpose() << endl;
      cout << "acc_avr: " << acc_avr.transpose() << endl;
      */

      /* state propagate */
      //cout << "dt_cov: " << dt_cov << endl;
      Predict(state_inout, dt_cov, true, true);
      StateEstimationIMU(state_inout);

      R_imu = state_inout.rot;
      acc_imu = R_imu * state_inout.acc + state_inout.gravity;
      pos_imu = state_inout.pos;
      vel_imu = state_inout.vel;
      time_offset = offs_t;
      
      /*
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt_cov);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt_cov);
      F_x.block<3, 3>(0, 10) = -Eye3d * dt_cov;
      F_x.block<3, 3>(3, 7) = Eye3d * dt_cov;
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt_cov;
      F_x.block<3, 3>(7, 13) = -R_imu * dt_cov;
      F_x.block<3, 3>(7, 16) = Eye3d * dt_cov;

      cov_w(6, 6) = cov_inv_expo * dt_cov * dt_cov;
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt_cov * dt_cov;
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt_cov * dt_cov;
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt_cov * dt_cov; // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt_cov * dt_cov; // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

      R_imu = R_imu * Exp_f;
      acc_imu = R_imu * acc_avr + state_inout.gravity;
      pos_imu = pos_imu + vel_imu * dt_cov + 0.5 * acc_imu * dt_cov * dt_cov;
      vel_imu = vel_imu + acc_imu * dt_cov;
      time_offset = offs_t;
      */

      /* save the poses at each IMU measurements */
      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      Pose6D head_pose = IMUpose.back();
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
      Pose6D tail_pose = IMUpose.back();

      pcl_body->clear();
      pcl_world->clear();
      M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
      V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
      /* deskew lidar points */
      if (lidar_meas.lio_vio_flg == LIO) {
        int cnt = 0;
        for (; it_pcl->curvature / double(1000) < tail_pose.offset_time; it_pcl++) {
          dt = it_pcl->curvature / double(1000) - head_pose.offset_time;

          M3D rot;
          V3D acc, pos, vel, gyr;
          rot << MAT_FROM_ARRAY(head_pose.rot);
          acc << VEC_FROM_ARRAY(head_pose.acc);
          vel << VEC_FROM_ARRAY(head_pose.vel);
          pos << VEC_FROM_ARRAY(head_pose.pos);
          gyr << VEC_FROM_ARRAY(head_pose.gyr);

          // Transform to the 'end' frame 
          M3D R_i(rot * Exp(gyr, dt));
          V3D T_i(pos + vel * dt + 0.5 * acc * dt * dt);
          V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
          V3D P_world = R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i;
          V3D P_compensate = extR_end * P_world + extT_end;
  
          /// save Undistorted points and their rotation
          PointType pt_b, pt_w;

          pt_b.x = P_compensate.x();
          pt_b.y = P_compensate.y();
          pt_b.z = P_compensate.z();

          pt_w.x = P_world.x();
          pt_w.y = P_world.y();
          pt_w.z = P_world.z();

          pcl_body->emplace_back(pt_b);
          pcl_world->emplace_back(pt_w);

          it_pcl->x = P_world(0);
          it_pcl->y = P_world(1);
          it_pcl->z = P_world(2);
          
          cnt ++;

          if (it_pcl == pcl_wait_proc.points.end() - 1) break;
        }
        cout << "number of points in imu frame: " << cnt << endl;

        voxelmap_manager->state_ = state_inout;
        voxelmap_manager->feats_down_body_ = pcl_body;
        voxelmap_manager->feats_down_world_ = pcl_world;
        voxelmap_manager->feats_down_size_ = cnt;
        voxelmap_manager->config_setting_.max_iterations_ = 1;

        voxelmap_manager->StateEstimation(state_inout);
        state_inout = voxelmap_manager->state_;

        R_imu = state_inout.rot;
        acc_imu = R_imu * state_inout.acc + state_inout.gravity;
        pos_imu = state_inout.pos;
        vel_imu = state_inout.vel;

        for (int i = 0; i < cnt; i++) {
          _pv_list.emplace_back(voxelmap_manager->pv_list_[i]);
        }
      }

      // what if stateEstimation is here?

      /*
      if (i == v_imu.size() - 2) {
        M3D rot;
        V3D acc, pos, vel, gyr;
        rot << MAT_FROM_ARRAY(tail_pose.rot);
        acc << VEC_FROM_ARRAY(tail_pose.acc);
        vel << VEC_FROM_ARRAY(tail_pose.vel);
        pos << VEC_FROM_ARRAY(tail_pose.pos);
        gyr << VEC_FROM_ARRAY(tail_pose.gyr);

        for (; it_pcl != pcl_wait_proc.points.end(); it_pcl++) {
          dt = it_pcl->curvature / double(1000) - tail_pose.offset_time;

          M3D R_i(rot * Exp(gyr, dt));
          V3D T_i(pos + vel * dt + 0.5 * acc * dt * dt);
          V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
          V3D P_world = R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i;
  
          /// save Undistorted points and their rotation
          //it_pcl->x = P_world(0);
          //it_pcl->y = P_world(1);
          //it_pcl->z = P_world(2);
          
          pcl_num ++;
        }
      }
      */
    }
    cout << IMUpose.size() - 1 << "-th time_offset:" << time_offset << endl;
    lidar_meas.last_lio_update_time = prop_end_time;
    break;
  }
  cout << "================== where are you? ==================" << endl;

  voxelmap_manager->config_setting_.max_iterations_ = max_iterations;
  /* state update */
  /*
  state_inout.vel = vel_imu;
  state_inout.rot = R_imu;
  state_inout.pos = pos_imu;
  */
  state_inout.inv_expo_time = tau;

  last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  if (pcl_wait_proc.points.size() < 1) return;
  
  if (lidar_meas.lio_vio_flg == LIO) {
    M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    
    it_pcl = pcl_wait_proc.points.begin();
    //for (; it_pcl->curvature / double(1000) < IMUpose.back().offset_time && it_pcl != pcl_wait_proc.points.end(); it_pcl++) {
    for (; it_pcl != pcl_wait_proc.points.end(); it_pcl++) {
      V3D p_world(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D p_body = extR_end * p_world + extT_end;

      it_pcl->x = p_body(0);
      it_pcl->y = p_body(1);
      it_pcl->z = p_body(2);
    }

    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
  
  
  /*
  if (lidar_meas.lio_vio_flg == LIO) {
    sort(pcl_wait_proc.points.begin(), pcl_wait_proc.points.end(), time_list);
    auto it_pcl = pcl_wait_proc.points.begin();
    M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto it_kp = IMUpose.begin(); it_kp != IMUpose.end() - 1; it_kp++) {
      auto head = it_kp;
      auto tail = it_kp + 1;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      int cnt = 0;
      for (; it_pcl->curvature / double(1000) < tail->offset_time; it_pcl++) {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        // Transform to the 'end' frame //
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_i(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt);
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        V3D P_world = R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i;
        V3D P_compensate = (extR_end * P_world + extT_end);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        cnt ++;
        if (it_pcl == pcl_wait_proc.points.end() - 1) break;
      }
      cout << "number of points in imu frame: " << cnt << endl;
    }
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
  */
}

void ImuProcess::UndistortPcl4(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out, 
                              vector<pointWithVar> &_pv_list, VoxelMapManagerPtr &voxelmap_manager) {
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  auto v_imu = meas.imu;
  //v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  std::vector<measure> meas_info;

  /*
  measure m;
  m.time = prop_beg_time;
  m.type = DEFAULT;
  meas_info.push_back(m);
  m.time = prop_end_time;
  m.type = DEFAULT;
  meas_info.push_back(m);
  */

  for (int i = 0; i < v_imu.size(); i++) {
    measure m;
    m.idx = i;
    m.time = v_imu[i]->header.stamp.toSec();
    m.type = IMU;
    //if (m.time > prop_beg_time) meas_info.push_back(m);
    meas_info.push_back(m);
  }

  if (lidar_meas.lio_vio_flg == LIO) {
    //downSizeFilterSurf.setInputCloud(lidar_meas.pcl_proc_cur);
    //downSizeFilterSurf.filter(pcl_wait_proc);
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;

    sort(pcl_wait_proc.begin(), pcl_wait_proc.end(), time_list);

    for (int i = 0; i < pcl_wait_proc.size(); i++) {
      measure m;
      m.idx = i;
      //m.time = pcl_wait_proc.points[i].curvature / double(1000) + lidar_meas.lidar_frame_beg_time;
      m.time = pcl_wait_proc.points[i].curvature / double(1000) + prop_beg_time;
      m.type = LIDAR;
      //if (m.time > prop_beg_time) meas_info.push_back(m);
      meas_info.push_back(m);
    }
  }
  sort(meas_info.begin(), meas_info.end());

  /*
  cout << fixed;
  cout << "meas_info.size(): " << meas_info.size() << endl;
  cout << "[ sensor time ]" << endl;
  cout << "prop_beg_time: " << prop_beg_time << endl;
  cout << "prop_end_time: " << prop_end_time << endl;
  cout << "last_prop_time: " << last_prop_time << endl;
  cout << "last_update_time: " << last_update_time << endl;
  */

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  M3D R_imu(state_inout.rot);
  double tau;
  if (!imu_time_init) {
    tau = 1.0;
    imu_time_init = true;
  }
  else {
    tau = state_inout.inv_expo_time;
  }

  switch (lidar_meas.lio_vio_flg) {
  case LIO:
  case VIO:
    for (int i = 0; i < meas_info.size(); i++) {
      auto &cur_meas = meas_info[i];

      if (i == 0) cout << "first_meas_time: " << cur_meas.time << endl;
      if (i == meas_info.size() - 1) cout << "last_meas_time: " << cur_meas.time << endl;

      if (cur_meas.type == IMU) {
        double dt;
        dt = cur_meas.time - last_update_time;

        auto head = v_imu[cur_meas.idx];

        angvel_avr << head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z;
        acc_avr << head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z;

        omg_meas = angvel_avr;
        acc_meas = acc_avr * G_m_s2 / mean_acc.norm();

        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

        Predict(state_inout, dt, true, true);
        StateEstimationIMU(state_inout);

        last_prop_time = cur_meas.time;
        last_update_time = cur_meas.time;
      } 
      else if (cur_meas.type == LIDAR) {
        if (cur_meas.idx == 0) cout << "first_lidar_time: " << cur_meas.time << endl;
        if (cur_meas.idx == pcl_wait_proc.size() - 1) cout << "last_lidar_time: " << cur_meas.time << endl;

        auto &pt = pcl_wait_proc.points[cur_meas.idx];
        double dt = cur_meas.time - last_update_time;

        M3D R_i(state_inout.rot * Exp(state_inout.omg, dt));
        V3D acc_imu = state_inout.rot * state_inout.acc + state_inout.gravity;
        V3D T_i(state_inout.pos + state_inout.vel * dt + 0.5 * acc_imu * dt * dt);
        V3D P_i(pt.x, pt.y, pt.z);
        V3D P_world(R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i);

        pt.x = P_world.x();
        pt.y = P_world.y();
        pt.z = P_world.z();
      } 
      else {
        double dt = cur_meas.time - last_update_time;

        Predict(state_inout, dt, true, true);

        last_prop_time = cur_meas.time;
        last_update_time = cur_meas.time;
      }
    }

    lidar_meas.last_lio_update_time = prop_end_time;
    break;
  }

  state_inout.inv_expo_time = tau;

  //last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO update ***/
  if (lidar_meas.lio_vio_flg == LIO) {
    M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto &pt: pcl_wait_proc) {
      V3D p(pt.x, pt.y, pt.z);
      V3D p_compensate(extR_end * p + extT_end);
      pt.x = p_compensate.x();
      pt.y = p_compensate.y();
      pt.z = p_compensate.z();
    }

    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
}

void ImuProcess::UndistortPcl5(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out, 
                               vector<pointWithVar> &_pv_list, VoxelMapManagerPtr &voxelmap_manager) {
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  auto v_imu = meas.imu;
  //v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;
  int effect_feat_num = 0;

  //last_prop_time = prop_beg_time;
  //last_update_time = prop_beg_time;

  cout << fixed;
  cout << "last_prop_time: " << last_prop_time << endl;
  cout << "last_update_time: " << last_update_time << endl;
  cout << "lidar_frame_beg_time: " << lidar_meas.lidar_frame_beg_time << endl;
  cout << "prop_beg_time: " << prop_beg_time << endl;
  cout << "prop_end_time: " << prop_end_time << endl;
  cout << "v_imu.size(): " << v_imu.size() << endl;

  std::vector<measure> meas_info;

  /*
  measure m;
  m.time = prop_beg_time;
  m.type = DEFAULT;
  meas_info.push_back(m);
  m.time = prop_end_time;
  m.type = DEFAULT;
  meas_info.push_back(m);
  */

  for (int i = 0; i < v_imu.size(); i++) {
    measure m;
    m.idx = i;
    m.time = v_imu[i]->header.stamp.toSec();
    m.type = IMU;
    //if (m.time > prop_beg_time) meas_info.push_back(m);
    meas_info.push_back(m);
  }

  if (lidar_meas.lio_vio_flg == LIO) {
    //pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    //pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    downSizeFilterSurf.setInputCloud(lidar_meas.pcl_proc_cur);
    downSizeFilterSurf.filter(pcl_wait_proc);
    lidar_meas.lidar_scan_index_now = 0;

    int feats_size = lidar_meas.pcl_proc_cur->size(); 

    // initialize voxelmap manager
    PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(pcl_wait_proc));
    int feats_down_size = feats_down_body->size();
    voxelmap_manager->feats_down_body_ = feats_down_body;
    voxelmap_manager->feats_down_size_ = feats_down_size;

    voxelmap_manager->feats_down_world_->resize(feats_down_size);
    voxelmap_manager->pv_list_.resize(feats_down_size);
    voxelmap_manager->cross_mat_list_.resize(feats_down_size);
    voxelmap_manager->body_cov_list_.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    cout << "feats size: " << feats_size << endl;
    cout << "feats down size: " << feats_down_size << endl;

    static int init_size = 0;
    if (init_size < 2000) {
      ivox = std::make_shared<IVoxType>(ivox_options);
      PointCloudXYZI::Ptr feats_init_world(new PointCloudXYZI());
      feats_init_world->resize(feats_size);

      for (int i = 0; i < feats_size; i ++) {
        auto &point_body = lidar_meas.pcl_proc_cur->points[i];
        auto &point_world = feats_init_world->points[i];

        V3D pt(point_body.x, point_body.y, point_body.z);
        V3D pt_w(state_inout.rot * (Lid_rot_to_IMU * pt + Lid_offset_to_IMU) + state_inout.pos);
        point_world.x = pt_w(0);
        point_world.y = pt_w(1);
        point_world.z = pt_w(2);
        point_world.intensity = point_body.intensity;
      }

      ivox->AddPoints(feats_init_world->points);
      init_size += feats_size;

      if (init_size >= 2000) {
        cout << "map initialized." << endl;
        cout << "ivox size: " << ivox->NumValidGrids() << endl;
      }
    }

    for (int i = 0; i < pcl_wait_proc.size(); i++) {
      measure m;
      m.idx = i;
      if (slam_mode == LIVO) m.time = pcl_wait_proc.points[i].curvature / double(1000) + prop_beg_time;
      else m.time = pcl_wait_proc.points[i].curvature / double(1000) + lidar_meas.lidar_frame_beg_time;
      m.type = LIDAR;
      meas_info.push_back(m);
    }
    //sort(pcl_wait_proc.points.begin(), pcl_wait_proc.points.end(), time_list);
    //cout << "last lidar time shoud be " << pcl_wait_proc.points.back().curvature / double(1000) + lidar_meas.lidar_frame_beg_time << endl;
  }
  sort(meas_info.begin(), meas_info.end());

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel), pos_imu(state_inout.pos);
  M3D R_imu(state_inout.rot);
  double offs_t;
  double tau;
  if (!imu_time_init) {
    tau = 1.0;
    imu_time_init = true;
  }
  else {
    tau = state_inout.inv_expo_time;
  }

  int imu_cnt = 0;
  int lid_cnt = 0;
  switch (lidar_meas.lio_vio_flg) {
    case LIO:
    case VIO:
    for (int i = 0; i < meas_info.size(); i++) {
      auto &cur_meas = meas_info[i];

      if (cur_meas.type == IMU) {
        auto head = v_imu[cur_meas.idx];

        if (imu_cnt == 0) cout << "first imu time: " << cur_meas.time << endl;
        if (imu_cnt == v_imu.size() - 1) cout << "last imu time: " << cur_meas.time << endl;
        imu_cnt ++;

        //angvel_avr << 0.5 * (head->angular_velocity.x + last_imu->angular_velocity.x), 
        //              0.5 * (head->angular_velocity.y + last_imu->angular_velocity.y),
        //              0.5 * (head->angular_velocity.z + last_imu->angular_velocity.z);

        //acc_avr << 0.5 * (head->linear_acceleration.x + last_imu->linear_acceleration.x), 
        //           0.5 * (head->linear_acceleration.y + last_imu->linear_acceleration.y),
        //           0.5 * (head->linear_acceleration.z + last_imu->linear_acceleration.z);

        angvel_avr << head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z;
        acc_avr << head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z;

        omg_meas = angvel_avr;
        acc_meas = acc_avr * G_m_s2 / mean_acc.norm();

        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

        double dt = cur_meas.time - last_prop_time;
        double dt_cov = cur_meas.time - last_update_time;

        // estimation
        Predict(state_inout, dt, true, false);
        Predict(state_inout, dt_cov, false, true);
        //Predict(state_inout, dt, true, true);
        StateEstimationIMU(state_inout);

        last_prop_time = cur_meas.time;
        last_update_time = cur_meas.time;

        last_imu = head;
      } 
      else if (cur_meas.type == LIDAR) {
        auto &pt = pcl_wait_proc.points[cur_meas.idx];

        if (lid_cnt == 0) cout << "first lidar time: " << cur_meas.time << endl;
        if (lid_cnt == pcl_wait_proc.size() - 1) cout << "last lidar time: " << cur_meas.time << endl;
        lid_cnt ++;

        double dt = cur_meas.time - last_prop_time;
        double dt_cov = cur_meas.time - last_update_time;

        M3D R_i;
        V3D T_i;

        Predict(state_inout, dt, true, false);
        //Predict(state_inout, dt, true, true);
        // estimation
        voxelmap_manager->state_ = state_inout;
        voxelmap_manager->StateEstimationCustom(state_inout, cur_meas.idx, 1, effect_feat_num);
        state_inout = voxelmap_manager->state_;

        last_prop_time = cur_meas.time;

        R_i = state_inout.rot;
        T_i = state_inout.pos;

        V3D P_i(pt.x, pt.y, pt.z);
        V3D P_world(R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_i);

        pt.x = P_world.x();
        pt.y = P_world.y();
        pt.z = P_world.z();
      }
      else {
        double dt = cur_meas.time - last_prop_time;
        double dt_cov = cur_meas.time - last_update_time;

        Predict(state_inout, dt, true, false);
        //Predict(state_inout, dt, true, true);

        last_prop_time = cur_meas.time;
        last_update_time = cur_meas.time;
      }
    }

    lidar_meas.last_lio_update_time = prop_end_time;
    break;
  }

  if (lidar_meas.lio_vio_flg == LIO) cout << "effect_feat_num: " << effect_feat_num << endl;
  state_inout.inv_expo_time = tau;

  //last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO update ***/
  if (lidar_meas.lio_vio_flg == LIO) {
    M3D extR_end(Lid_rot_to_IMU.transpose() * state_inout.rot.transpose());
    V3D extT_end(-extR_end * state_inout.pos - Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto &pt: pcl_wait_proc) {
      V3D p(pt.x, pt.y, pt.z);
      V3D p_compensate(extR_end * p + extT_end);
      pt.x = p_compensate.x();
      pt.y = p_compensate.y();
      pt.z = p_compensate.z();
    }

    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    //IMUpose.clear();
  }

  //state_inout = state_inout_cp;
}

void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_, 
                          vector<pointWithVar> &_pv_list, VoxelMapManagerPtr &voxelmap_manager) {
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  if (!imu_en)
  {
    Forward_without_imu(lidar_meas, stat, *cur_pcl_un_);
    return;
  }

  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init)
  {
    double pcl_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;
    // lidar_meas.last_lio_update_time = pcl_end_time;

    if (meas.imu.empty()) { return; };
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init = true;

    last_imu = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      // cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init = false;
      imu_need_init = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: "
               "%.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f \n",
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
               cov_gyr[2]);
      ROS_INFO("IMU Initials: ba covarience: %.8f %.8f %.8f; bg covarience: "
               "%.8f %.8f %.8f",
               cov_bias_acc[0], cov_bias_acc[1], cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }

  /*
  cout << "cov inverse before propagation:\n" << endl;
  auto cov1 = stat.cov;
  cov1 = cov1.inverse();
  cout << cov1.block<6, 6>(0, 0) << endl << endl;
  */
  //cout << cov1.block<6, 6>(6, 6) << endl << endl;
  //cout << cov1.block<6, 6>(12, 12) << endl << endl;

  //UndistortPcl(lidar_meas, stat, *cur_pcl_un_);
  //UndistortPclPointLIO(lidar_meas, stat, *cur_pcl_un_);
  //UndistortPcl2(lidar_meas, stat, *cur_pcl_un_);
  //UndistortPcl3(lidar_meas, stat, *cur_pcl_un_, _pv_list, voxelmap_manager);
  //UndistortPcl4(lidar_meas, stat, *cur_pcl_un_, _pv_list, voxelmap_manager);
  UndistortPcl5(lidar_meas, stat, *cur_pcl_un_, _pv_list, voxelmap_manager);
  // cout << "[ IMU ] undistorted point num: " << cur_pcl_un_->size() << endl;

  /*
  cout << "cov inverse after propagation:\n" << endl;
  auto cov2 = stat.cov;
  cov2 = cov2.inverse();
  cout << cov2.block<6, 6>(0, 0) << endl << endl;
  */
  //cout << cov2.block<6, 6>(6, 6) << endl << endl;
  //cout << cov2.block<6, 6>(12, 12) << endl << endl;
}

void ImuProcess::Predict(StatesGroup &stat, double dt, bool predict_state, bool prop_cov) {
  V3D omg = stat.omg;              // imu measurement model
  V3D acc = stat.acc;              // imu measurement model
  //V3D omg = omg_meas - stat.bias_g;  // imu input model
  //V3D acc = acc_meas - stat.bias_a;  // imu input model
  
  if (prop_cov) {
    MD(DIM_STATE, DIM_STATE) F_x;
    M3D acc_skew;
    acc_skew << SKEW_SYM_MATRX(acc);

    F_x.setIdentity();
    F_x.block<3, 3>(0, 0) = Exp(omg, -dt);

    F_x.block<3, 3>(0, 19) = M3D::Identity() * dt;     // imu measurement model
    //F_x.block<3, 3>(0, 10) = -M3D::Identity() * dt;    // imu input model

    F_x.block<3, 3>(3, 7) = M3D::Identity() * dt;
    F_x.block<3, 3>(7, 0) = -stat.rot * acc_skew * dt;
    F_x.block<3, 3>(7, 16) = M3D::Identity() * dt;

    F_x.block<3, 3>(7, 22) = stat.rot * dt;            // imu measurement model
    //F_x.block<3, 3>(7, 13) = -stat.rot * dt;           // imu input model
 
    stat.cov = F_x * stat.cov * F_x.transpose() + cov_w * dt * dt;
  }

  if (predict_state) {
    /*
    MD(DIM_STATE, 1) f;
    V3D acc_imu = stat.rot * acc + stat.gravity;
    f.setZero();
    f.block<3, 1>(0, 0) = omg;                                   // rot 
    f.block<3, 1>(3, 0) = stat.vel + 0.5 * acc_imu * dt;         // pos
    f.block<3, 1>(7, 0) = acc_imu;                               // vel
    stat += f * dt;
    */
    stat.rot = stat.rot * Exp(omg, dt);
    V3D acc_imu = stat.rot * acc + stat.gravity;
    stat.pos += stat.vel * dt + 0.5 * acc_imu * dt * dt;
    stat.vel += acc_imu * dt;
  }
}

void ImuProcess::StateEstimationIMU(StatesGroup &stat) {  
  for (int k = 0; k < 3; k++) {
    if (omg_meas[k] > 0.99 * satu_gyr) omg_meas[k] = 0.0;
    if (acc_meas[k] > 0.99 * satu_acc) acc_meas[k] = 0.0;
  }
  
  MD(6, 1) z;
  z.block<3, 1>(0, 0) = omg_meas - stat.bias_g - stat.omg;
  z.block<3, 1>(3, 0) = acc_meas - stat.bias_a - stat.acc;

  MD(DIM_STATE, 6) PHT;
  MD(6, DIM_STATE) HP;
  MD(6, 6) HPHT;
  PHT.setZero();
  HP.setZero();
  HPHT.setZero();

  for (int k = 0; k < 6; k++) {
    PHT.col(k) = stat.cov.col(10 + k) + stat.cov.col(19 + k);
    HP.row(k) = stat.cov.row(10 + k) + stat.cov.row(19 + k);
  }
  for (int k = 0; k < 6; k++) {
    HPHT.col(k) = HP.col(10 + k) + HP.col(19 + k);
  }
  HPHT += R_IMU;

  MD(DIM_STATE, 6) K = PHT * HPHT.inverse();
  MD(DIM_STATE, 1) dx = K * z;

  //cout << "stat.cov.block<6, 6>(10, 10):\n" << stat.cov.block<6, 6>(10, 10) << endl;
  //cout << "stat.cov.block<6, 6>(19, 19):\n" << stat.cov.block<6, 6>(19, 19) << endl;
  //cout << "PHT:\n" << PHT << endl;
  //cout << "dx: " << dx.transpose() << endl;

  stat.cov -= K * HP;
  stat += dx;
}