/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/
// Modified in this fork by <lUCAS> on 2026-04-23.
// Summary of changes: ROS 2 Jazzy compatibility, logging, and research-related extensions.
// This file remains part of a GPL-3.0-licensed derivative work.
// See the repository root LICENSE and THIRD_PARTY_NOTICES.md.

#include "parameters.h"
#include <regex>
#include <fstream>
#include <sstream>
#include <algorithm>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string IMU_NOISE_MODE;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

static bool parseVehicleTxtToT(const std::string &path, Eigen::Matrix4d &T);
static bool parseRightYamlBaseline(const std::string &path, double &baseline);
static bool inferSequenceFromOutputPath(const std::string &output_path, std::string &seq_name);
static bool loadSequenceSpecificExtrinsic(
    const std::string &output_folder,
    Eigen::Matrix4d &T_b_c0,
    Eigen::Matrix4d &T_b_c1);

template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

static void applyImuNoiseProfile(const std::string &mode, cv::FileStorage &fsSettings)
{
    if (mode == "xsens")
    {
        ACC_N = 0.1;
        ACC_W = 0.001;
        GYR_N = 0.01;
        GYR_W = 0.0001;
        ROS_WARN("Use built-in IMU noise profile: xsens");
    }
    else if (mode == "fog_xsens")
    {
        ACC_N = 0.1;
        ACC_W = 0.001;
        GYR_N = 0.001;
        GYR_W = 0.00001;
        ROS_WARN("Use built-in IMU noise profile: fog_xsens");
    }
    else if (mode == "custom")
    {
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        ROS_WARN("Use custom IMU noise profile from YAML");
    }
    else
    {
        ROS_WARN("Unknown imu_noise_mode=%s, fallback to custom", mode.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
    }
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());

        if (!fsSettings["imu_noise_mode"].empty())
            fsSettings["imu_noise_mode"] >> IMU_NOISE_MODE;
        else
            IMU_NOISE_MODE = "custom";

        applyImuNoiseProfile(IMU_NOISE_MODE, fsSettings);

        G.z() = fsSettings["g_norm"];

        std::cout << "IMU_NOISE_MODE: " << IMU_NOISE_MODE << std::endl;
        std::cout << "ACC_N: " << ACC_N << " ACC_W: " << ACC_W
                << " GYR_N: " << GYR_N << " GYR_W: " << GYR_W << std::endl;
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";

    Eigen::Matrix4d T0_seq, T1_seq;
    bool loaded_seq_ex = loadSequenceSpecificExtrinsic(OUTPUT_FOLDER, T0_seq, T1_seq);

    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        if (loaded_seq_ex)
        {
            ROS_WARN("Use sequence-specific extrinsic for cam0 from calibration folder");
            RIC.push_back(T0_seq.block<3, 3>(0, 0));
            TIC.push_back(T0_seq.block<3, 1>(0, 3));
        }
        else
        {
            ROS_WARN("Fallback to YAML body_T_cam0");
            cv::Mat cv_T;
            fsSettings["body_T_cam0"] >> cv_T;
            Eigen::Matrix4d T;
            cv::cv2eigen(cv_T, T);
            RIC.push_back(T.block<3, 3>(0, 0));
            TIC.push_back(T.block<3, 1>(0, 3));
        }
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        if (loaded_seq_ex)
        {
            ROS_WARN("Use sequence-specific extrinsic for cam1 from calibration folder");
            RIC.push_back(T1_seq.block<3, 3>(0, 0));
            TIC.push_back(T1_seq.block<3, 1>(0, 3));
        }
        else
        {
            ROS_WARN("Fallback to YAML body_T_cam1");
            cv::Mat cv_T;
            fsSettings["body_T_cam1"] >> cv_T;
            Eigen::Matrix4d T;
            cv::cv2eigen(cv_T, T);
            RIC.push_back(T.block<3, 3>(0, 0));
            TIC.push_back(T.block<3, 1>(0, 3));
        }
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}


/////////my
static bool parseVehicleTxtToT(const std::string &path, Eigen::Matrix4d &T)
{
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::stringstream buffer;
    buffer << fin.rdbuf();
    std::string text = buffer.str();

    auto parse_numbers = [](const std::string &s) {
        std::vector<double> nums;
        std::regex num_re(R"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)");
        auto begin = std::sregex_iterator(s.begin(), s.end(), num_re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) nums.push_back(std::stod(it->str()));
        return nums;
    };

    std::regex rR(R"(R\s*:\s*([^\n\r]+))");
    std::regex rT(R"(T\s*:\s*([^\n\r]+))");
    std::smatch mR, mT;
    if (std::regex_search(text, mR, rR) && std::regex_search(text, mT, rT))
    {
        auto Rnums = parse_numbers(mR[1].str());
        auto Tnums = parse_numbers(mT[1].str());
        if (Rnums.size() == 9 && Tnums.size() == 3)
        {
            T.setIdentity();
            T(0,0)=Rnums[0]; T(0,1)=Rnums[1]; T(0,2)=Rnums[2];
            T(1,0)=Rnums[3]; T(1,1)=Rnums[4]; T(1,2)=Rnums[5];
            T(2,0)=Rnums[6]; T(2,1)=Rnums[7]; T(2,2)=Rnums[8];
            T(0,3)=Tnums[0]; T(1,3)=Tnums[1]; T(2,3)=Tnums[2];
            return true;
        }
    }

    auto nums = parse_numbers(text);
    if (nums.size() == 15)
    {
        T.setIdentity();
        T(0,0)=nums[3];  T(0,1)=nums[4];  T(0,2)=nums[5];
        T(1,0)=nums[6];  T(1,1)=nums[7];  T(1,2)=nums[8];
        T(2,0)=nums[9];  T(2,1)=nums[10]; T(2,2)=nums[11];
        T(0,3)=nums[12]; T(1,3)=nums[13]; T(2,3)=nums[14];
        return true;
    }

    return false;
}

static bool parseRightYamlBaseline(const std::string &path, double &baseline)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    cv::Mat P;
    fs["projection_matrix"] >> P;
    fs.release();
    if (P.rows != 3 || P.cols != 4) return false;
    double fx = P.at<double>(0,0);
    double tx = P.at<double>(0,3);
    baseline = -tx / fx;
    return true;
}

static bool inferSequenceFromOutputPath(const std::string &output_path, std::string &seq_name)
{
    std::regex re(R"(urban(\d+)[_-]([A-Za-z0-9]+))", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(output_path, m, re)) return false;
    seq_name = "urban" + m[1].str() + "-" + m[2].str();
    std::transform(seq_name.begin(), seq_name.end(), seq_name.begin(), ::tolower);
    return true;
}

static bool loadSequenceSpecificExtrinsic(
    const std::string &output_folder,
    Eigen::Matrix4d &T_b_c0,
    Eigen::Matrix4d &T_b_c1)
{
    std::string seq_name;
    if (!inferSequenceFromOutputPath(output_folder, seq_name))
        return false;

    const std::string dataset_root = "/mnt/sata4t/datasets/kaist_complex_urban/extracted";
    const std::string calib_dir = dataset_root + "/" + seq_name + "/calibration/" + seq_name + "/calibration";

    Eigen::Matrix4d T_v_i, T_v_s;
    if (!parseVehicleTxtToT(calib_dir + "/Vehicle2IMU.txt", T_v_i)) return false;
    if (!parseVehicleTxtToT(calib_dir + "/Vehicle2Stereo.txt", T_v_s)) return false;

    double baseline = 0.0;
    if (!parseRightYamlBaseline(calib_dir + "/right.yaml", baseline)) return false;

    T_b_c0 = T_v_i.inverse() * T_v_s;

    Eigen::Matrix4d T_c0_c1 = Eigen::Matrix4d::Identity();
    T_c0_c1(0,3) = baseline;
    T_b_c1 = T_b_c0 * T_c0_c1;
    return true;
}