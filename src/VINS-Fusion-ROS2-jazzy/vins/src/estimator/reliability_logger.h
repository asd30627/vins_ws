#pragma once

#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <filesystem>

struct ReliabilityCsvRow
{
    std::string run_id;
    std::string dataset_name;
    std::string sequence_name;

    long long update_id = 0;
    int frame_count = 0;
    double timestamp = 0.0;

    std::string solver_flag;
    int use_imu = 0;
    int stereo = 0;
    int current_is_keyframe = 0;

    double feature_tracker_time_ms = 0.0;
    int tracked_feature_count_raw = 0;
    int tracked_feature_count_mgr = 0;

    double mean_track_vel_px = 0.0;
    double median_track_vel_px = 0.0;
    double coverage_4x4 = 0.0;
    double img_dt_sec = 0.0;

    double solver_time_ms_last = 0.0;
    int outlier_count_last = 0;
    int inlier_count_last = 0;
    double outlier_ratio_last = 0.0;
    int failure_detected_last = 0;

    double est_p_x = 0.0;
    double est_p_y = 0.0;
    double est_p_z = 0.0;

    double est_q_x = 0.0;
    double est_q_y = 0.0;
    double est_q_z = 0.0;
    double est_q_w = 1.0;

    double visual_w_pred = 1.0;
    int visual_gate_pass = 1;
    double visual_alpha = 1.0;
    int visual_has_prediction = 0;

    int imu_sample_count = 0;
    double acc_norm_mean = 0.0;
    double gyr_norm_mean = 0.0;

    double avg_track_length = 0.0;
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

        // 如果已經開著同一個檔案，就不要再重開、更不要 truncate
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

        // 重點：改成 app，不要每次 open 都 trunc
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

        ofs_ << csvSafe(r.run_id) << ","
             << csvSafe(r.dataset_name) << ","
             << csvSafe(r.sequence_name) << ","
             << r.update_id << ","
             << r.frame_count << ","
             << std::fixed << std::setprecision(9) << r.timestamp << ","
             << csvSafe(r.solver_flag) << ","
             << r.use_imu << ","
             << r.stereo << ","
             << r.current_is_keyframe << ","
             << std::setprecision(6)
             << r.feature_tracker_time_ms << ","
             << r.tracked_feature_count_raw << ","
             << r.tracked_feature_count_mgr << ","
             << r.mean_track_vel_px << ","
             << r.median_track_vel_px << ","
             << r.coverage_4x4 << ","
             << r.img_dt_sec << ","
             << r.solver_time_ms_last << ","
             << r.outlier_count_last << ","
             << r.inlier_count_last << ","
             << r.outlier_ratio_last << ","
             << r.failure_detected_last << ","
             << r.est_p_x << ","
             << r.est_p_y << ","
             << r.est_p_z << ","
             << r.est_q_x << ","
             << r.est_q_y << ","
             << r.est_q_z << ","
             << r.est_q_w << ","
             << r.visual_w_pred << ","
             << r.visual_gate_pass << ","
             << r.visual_alpha << ","
             << r.visual_has_prediction << ","
             << r.imu_sample_count << ","
             << r.acc_norm_mean << ","
             << r.gyr_norm_mean << ","
             << r.avg_track_length
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
        ofs_ << "run_id,"
             << "dataset_name,"
             << "sequence_name,"
             << "update_id,"
             << "frame_count,"
             << "timestamp,"
             << "solver_flag,"
             << "use_imu,"
             << "stereo,"
             << "current_is_keyframe,"
             << "feature_tracker_time_ms,"
             << "tracked_feature_count_raw,"
             << "tracked_feature_count_mgr,"
             << "mean_track_vel_px,"
             << "median_track_vel_px,"
             << "coverage_4x4,"
             << "img_dt_sec,"
             << "solver_time_ms_last,"
             << "outlier_count_last,"
             << "inlier_count_last,"
             << "outlier_ratio_last,"
             << "failure_detected_last,"
             << "est_p_x,"
             << "est_p_y,"
             << "est_p_z,"
             << "est_q_x,"
             << "est_q_y,"
             << "est_q_z,"
             << "est_q_w,"
             << "visual_w_pred,"
             << "visual_gate_pass,"
             << "visual_alpha,"
             << "visual_has_prediction,"
             << "imu_sample_count,"
             << "acc_norm_mean,"
             << "gyr_norm_mean,"
             << "avg_track_length"
             << "\n";
        ofs_.flush();
    }
};