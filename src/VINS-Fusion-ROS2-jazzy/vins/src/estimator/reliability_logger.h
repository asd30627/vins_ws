#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>

struct ReliabilityCsvRow
{
    // ===== identity / timing =====
    int schema_version = 2;
    std::string run_id;
    std::string dataset_name;
    std::string sequence_name;
    long long update_id = 0;
    int frame_count = 0;
    double timestamp = 0.0;

    // ===== estimator mode =====
    std::string solver_flag;
    int use_imu = 0;
    int stereo = 0;
    int current_is_keyframe = 0;
    int estimate_extrinsic = 0;
    int estimate_td = 0;
    double td_current = 0.0;

    // ===== frontend / tracker quality =====
    double feature_tracker_time_ms = 0.0;
    int tracked_feature_count_raw = 0;
    int tracked_feature_count_mgr = 0;

    double mean_track_vel_px = 0.0;
    double median_track_vel_px = 0.0;
    double min_track_vel_px = 0.0;
    double max_track_vel_px = 0.0;
    double std_track_vel_px = 0.0;
    double p90_track_vel_px = 0.0;

    double coverage_4x4 = 0.0;
    double coverage_8x8 = 0.0;
    int occupied_cells_4x4 = 0;
    int occupied_cells_8x8 = 0;
    double feature_entropy_4x4 = 0.0;
    double feature_entropy_8x8 = 0.0;

    double img_dt_sec = 0.0;

    // ===== backend health =====
    double solver_time_ms_last = 0.0;
    int outlier_count_last = 0;
    int inlier_count_last = 0;
    double outlier_ratio_last = 0.0;
    int failure_detected_last = 0;
    std::string failure_reason_proxy = "none";

    // ===== current state =====
    double est_p_x = 0.0;
    double est_p_y = 0.0;
    double est_p_z = 0.0;
    double est_q_x = 0.0;
    double est_q_y = 0.0;
    double est_q_z = 0.0;
    double est_q_w = 1.0;

    double vel_norm = 0.0;
    double ba_norm = 0.0;
    double bg_norm = 0.0;

    double delta_p_norm = 0.0;
    double delta_q_deg = 0.0;

    // ===== visual admission state =====
    double visual_w_pred = 1.0;
    int visual_gate_pass = 1;
    double visual_alpha = 1.0;
    int visual_has_prediction = 0;
    double visual_soft_target_proxy = -1.0;

    // ===== IMU statistics =====
    int imu_sample_count = 0;
    double acc_norm_mean = 0.0;
    double acc_norm_std = 0.0;
    double acc_norm_max = 0.0;
    double gyr_norm_mean = 0.0;
    double gyr_norm_std = 0.0;
    double gyr_norm_max = 0.0;

    // ===== FeatureManager / geometry statistics =====
    double avg_track_length = 0.0;
    double track_len_min = 0.0;
    double track_len_max = 0.0;
    double track_len_std = 0.0;
    double track_len_p90 = 0.0;

    int good_depth_count = 0;
    int bad_depth_count = 0;
    double depth_mean = 0.0;
    double depth_min = 0.0;
    double depth_max = 0.0;
    double depth_std = 0.0;

    // ===== extrinsic telemetry: cam0 =====
    double cam0_tic_x = 0.0;
    double cam0_tic_y = 0.0;
    double cam0_tic_z = 0.0;
    double cam0_q_x = 0.0;
    double cam0_q_y = 0.0;
    double cam0_q_z = 0.0;
    double cam0_q_w = 1.0;
    double cam0_init_delta_t_norm = 0.0;
    double cam0_init_delta_r_deg = 0.0;

    // ===== extrinsic telemetry: cam1 =====
    double cam1_tic_x = 0.0;
    double cam1_tic_y = 0.0;
    double cam1_tic_z = 0.0;
    double cam1_q_x = 0.0;
    double cam1_q_y = 0.0;
    double cam1_q_z = 0.0;
    double cam1_q_w = 1.0;
    double cam1_init_delta_t_norm = 0.0;
    double cam1_init_delta_r_deg = 0.0;
};

class ReliabilityCsvLogger
{
public:
    ReliabilityCsvLogger() = default;

    ~ReliabilityCsvLogger()
    {
        close();
    }

    void open(const std::string &path)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (ofs_.is_open() && path_ == path)
            return;

        if (ofs_.is_open())
            ofs_.close();

        path_ = path;

        std::filesystem::path p(path_);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        const bool need_header =
            (!std::filesystem::exists(path_)) ||
            (std::filesystem::exists(path_) && std::filesystem::file_size(path_) == 0);

        ofs_.open(path_, std::ios::out | std::ios::app);
        if (!ofs_.is_open())
            throw std::runtime_error("Failed to open CSV file: " + path_);

        if (need_header)
            writeHeader();

        ofs_.flush();
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (ofs_.is_open())
            ofs_.close();
    }

    bool isOpen() const
    {
        return ofs_.is_open();
    }

    void append(const ReliabilityCsvRow &r)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!ofs_.is_open())
            return;

        ofs_
            << r.schema_version << ","
            << csvSafe(r.run_id) << ","
            << csvSafe(r.dataset_name) << ","
            << csvSafe(r.sequence_name) << ","
            << r.update_id << ","
            << r.frame_count << ","
            << std::fixed << std::setprecision(9)
            << r.timestamp << ","

            << csvSafe(r.solver_flag) << ","
            << r.use_imu << ","
            << r.stereo << ","
            << r.current_is_keyframe << ","
            << r.estimate_extrinsic << ","
            << r.estimate_td << ","
            << std::setprecision(9)
            << r.td_current << ","

            << std::setprecision(6)
            << r.feature_tracker_time_ms << ","
            << r.tracked_feature_count_raw << ","
            << r.tracked_feature_count_mgr << ","

            << r.mean_track_vel_px << ","
            << r.median_track_vel_px << ","
            << r.min_track_vel_px << ","
            << r.max_track_vel_px << ","
            << r.std_track_vel_px << ","
            << r.p90_track_vel_px << ","

            << r.coverage_4x4 << ","
            << r.coverage_8x8 << ","
            << r.occupied_cells_4x4 << ","
            << r.occupied_cells_8x8 << ","
            << r.feature_entropy_4x4 << ","
            << r.feature_entropy_8x8 << ","

            << r.img_dt_sec << ","

            << r.solver_time_ms_last << ","
            << r.outlier_count_last << ","
            << r.inlier_count_last << ","
            << r.outlier_ratio_last << ","
            << r.failure_detected_last << ","
            << csvSafe(r.failure_reason_proxy) << ","

            << r.est_p_x << ","
            << r.est_p_y << ","
            << r.est_p_z << ","
            << r.est_q_x << ","
            << r.est_q_y << ","
            << r.est_q_z << ","
            << r.est_q_w << ","

            << r.vel_norm << ","
            << r.ba_norm << ","
            << r.bg_norm << ","
            << r.delta_p_norm << ","
            << r.delta_q_deg << ","

            << r.visual_w_pred << ","
            << r.visual_gate_pass << ","
            << r.visual_alpha << ","
            << r.visual_has_prediction << ","
            << r.visual_soft_target_proxy << ","

            << r.imu_sample_count << ","
            << r.acc_norm_mean << ","
            << r.acc_norm_std << ","
            << r.acc_norm_max << ","
            << r.gyr_norm_mean << ","
            << r.gyr_norm_std << ","
            << r.gyr_norm_max << ","

            << r.avg_track_length << ","
            << r.track_len_min << ","
            << r.track_len_max << ","
            << r.track_len_std << ","
            << r.track_len_p90 << ","

            << r.good_depth_count << ","
            << r.bad_depth_count << ","
            << r.depth_mean << ","
            << r.depth_min << ","
            << r.depth_max << ","
            << r.depth_std << ","

            << r.cam0_tic_x << ","
            << r.cam0_tic_y << ","
            << r.cam0_tic_z << ","
            << r.cam0_q_x << ","
            << r.cam0_q_y << ","
            << r.cam0_q_z << ","
            << r.cam0_q_w << ","
            << r.cam0_init_delta_t_norm << ","
            << r.cam0_init_delta_r_deg << ","

            << r.cam1_tic_x << ","
            << r.cam1_tic_y << ","
            << r.cam1_tic_z << ","
            << r.cam1_q_x << ","
            << r.cam1_q_y << ","
            << r.cam1_q_z << ","
            << r.cam1_q_w << ","
            << r.cam1_init_delta_t_norm << ","
            << r.cam1_init_delta_r_deg
            << "\n";

        ofs_.flush();
    }

private:
    std::string path_;
    std::ofstream ofs_;
    mutable std::mutex mtx_;

    static std::string csvSafe(const std::string &s)
    {
        std::string out = s;
        for (char &c : out)
        {
            if (c == ',' || c == '\n' || c == '\r')
                c = '_';
        }
        return out;
    }

    void writeHeader()
    {
        ofs_
            << "schema_version,"
            << "run_id,"
            << "dataset_name,"
            << "sequence_name,"
            << "update_id,"
            << "frame_count,"
            << "timestamp,"

            << "solver_flag,"
            << "use_imu,"
            << "stereo,"
            << "current_is_keyframe,"
            << "estimate_extrinsic,"
            << "estimate_td,"
            << "td_current,"

            << "feature_tracker_time_ms,"
            << "tracked_feature_count_raw,"
            << "tracked_feature_count_mgr,"

            << "mean_track_vel_px,"
            << "median_track_vel_px,"
            << "min_track_vel_px,"
            << "max_track_vel_px,"
            << "std_track_vel_px,"
            << "p90_track_vel_px,"

            << "coverage_4x4,"
            << "coverage_8x8,"
            << "occupied_cells_4x4,"
            << "occupied_cells_8x8,"
            << "feature_entropy_4x4,"
            << "feature_entropy_8x8,"

            << "img_dt_sec,"

            << "solver_time_ms_last,"
            << "outlier_count_last,"
            << "inlier_count_last,"
            << "outlier_ratio_last,"
            << "failure_detected_last,"
            << "failure_reason_proxy,"

            << "est_p_x,"
            << "est_p_y,"
            << "est_p_z,"
            << "est_q_x,"
            << "est_q_y,"
            << "est_q_z,"
            << "est_q_w,"

            << "vel_norm,"
            << "ba_norm,"
            << "bg_norm,"
            << "delta_p_norm,"
            << "delta_q_deg,"

            << "visual_w_pred,"
            << "visual_gate_pass,"
            << "visual_alpha,"
            << "visual_has_prediction,"
            << "visual_soft_target_proxy,"

            << "imu_sample_count,"
            << "acc_norm_mean,"
            << "acc_norm_std,"
            << "acc_norm_max,"
            << "gyr_norm_mean,"
            << "gyr_norm_std,"
            << "gyr_norm_max,"

            << "avg_track_length,"
            << "track_len_min,"
            << "track_len_max,"
            << "track_len_std,"
            << "track_len_p90,"

            << "good_depth_count,"
            << "bad_depth_count,"
            << "depth_mean,"
            << "depth_min,"
            << "depth_max,"
            << "depth_std,"

            << "cam0_tic_x,"
            << "cam0_tic_y,"
            << "cam0_tic_z,"
            << "cam0_q_x,"
            << "cam0_q_y,"
            << "cam0_q_z,"
            << "cam0_q_w,"
            << "cam0_init_delta_t_norm,"
            << "cam0_init_delta_r_deg,"

            << "cam1_tic_x,"
            << "cam1_tic_y,"
            << "cam1_tic_z,"
            << "cam1_q_x,"
            << "cam1_q_y,"
            << "cam1_q_z,"
            << "cam1_q_w,"
            << "cam1_init_delta_t_norm,"
            << "cam1_init_delta_r_deg"
            << "\n";

        ofs_.flush();
    }
};