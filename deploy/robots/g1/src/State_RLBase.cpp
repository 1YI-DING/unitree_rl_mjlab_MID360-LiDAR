#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/utils/utils.h"
#include "unitree/dds_wrapper/common/Subscription.h"
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/PointField_.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>
#include <unordered_map>

namespace
{

using PointCloudMsg = sensor_msgs::msg::dds_::PointCloud2_;
using PointFieldMsg = sensor_msgs::msg::dds_::PointField_;

enum class HeightScanMode
{
    Normal,
    Zero,
    MissValue,
};

struct PointFieldOffsets
{
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

Eigen::Vector3f parse_vec3(const YAML::Node & node, const Eigen::Vector3f & fallback)
{
    if (!node || !node.IsSequence() || node.size() != 3) {
        return fallback;
    }

    return Eigen::Vector3f(
        node[0].as<float>(),
        node[1].as<float>(),
        node[2].as<float>()
    );
}

Eigen::Quaternionf quat_from_rpy_deg(const Eigen::Vector3f & rpy_deg)
{
    constexpr float kDegToRad = 0.01745329251994329577f;
    const Eigen::Vector3f rpy_rad = rpy_deg * kDegToRad;
    const Eigen::AngleAxisf roll(rpy_rad.x(), Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch(rpy_rad.y(), Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf yaw(rpy_rad.z(), Eigen::Vector3f::UnitZ());
    return (yaw * pitch * roll).normalized();
}

std::optional<PointFieldOffsets> point_field_offsets(const PointCloudMsg & msg)
{
    std::optional<uint32_t> x_offset;
    std::optional<uint32_t> y_offset;
    std::optional<uint32_t> z_offset;

    for (const auto & field : msg.fields()) {
        if (field.datatype() != sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_) {
            continue;
        }
        if (field.name() == "x") {
            x_offset = field.offset();
        } else if (field.name() == "y") {
            y_offset = field.offset();
        } else if (field.name() == "z") {
            z_offset = field.offset();
        }
    }

    if (!x_offset || !y_offset || !z_offset) {
        return std::nullopt;
    }

    return PointFieldOffsets{*x_offset, *y_offset, *z_offset};
}

float read_float32(const std::vector<uint8_t> & data, size_t offset)
{
    float value = 0.0f;
    std::memcpy(&value, data.data() + offset, sizeof(float));
    return value;
}

HeightScanMode parse_height_scan_mode(const std::string & mode)
{
    if (mode == "normal") {
        return HeightScanMode::Normal;
    }
    if (mode == "zero") {
        return HeightScanMode::Zero;
    }
    if (mode == "miss_value") {
        return HeightScanMode::MissValue;
    }
    spdlog::warn(
        "Unknown height_scan_mode '{}', falling back to 'normal'",
        mode
    );
    return HeightScanMode::Normal;
}

const char * height_scan_mode_name(const HeightScanMode mode)
{
    switch (mode) {
        case HeightScanMode::Normal:
            return "normal";
        case HeightScanMode::Zero:
            return "zero";
        case HeightScanMode::MissValue:
            return "miss_value";
    }
    return "normal";
}

}  // namespace

class LidarHeightScanUpdater
{
public:
    LidarHeightScanUpdater(
        isaaclab::Articulation * robot,
        const YAML::Node & state_cfg,
        const YAML::Node & env_cfg
    )
    : robot_(robot)
    {
        const YAML::Node lidar_cfg = state_cfg["lidar"];
        topic_ = lidar_cfg["topic"].as<std::string>("utlidar/cloud");
        processing_rate_hz_ = lidar_cfg["processing_rate_hz"].as<float>(20.0f);
        pointcloud_timeout_ms_ = lidar_cfg["pointcloud_timeout_ms"].as<int>(500);
        height_scan_mode_ = parse_height_scan_mode(
            lidar_cfg["height_scan_mode"].as<std::string>("normal")
        );

        const YAML::Node debug_cfg = lidar_cfg["debug_stats"];
        debug_stats_enabled_ = debug_cfg["enabled"].as<bool>(false);
        debug_stats_period_s_ = debug_cfg["period_s"].as<float>(1.0f);

        const YAML::Node extrinsics_cfg = lidar_cfg["extrinsics"];
        lidar_translation_in_pelvis_ = parse_vec3(
            extrinsics_cfg["translation_m"], Eigen::Vector3f::Zero()
        );
        lidar_rotation_in_pelvis_ = quat_from_rpy_deg(
            parse_vec3(extrinsics_cfg["rpy_deg"], Eigen::Vector3f::Zero())
        );

        const YAML::Node scan_cfg = lidar_cfg["height_scan"];
        size_x_m_ = scan_cfg["size_x_m"].as<float>(1.6f);
        size_y_m_ = scan_cfg["size_y_m"].as<float>(1.0f);
        resolution_m_ = scan_cfg["resolution_m"].as<float>(0.1f);
        miss_value_m_ = scan_cfg["miss_value_m"].as<float>(5.0f);
        min_point_z_m_ = scan_cfg["min_point_z_m"].as<float>(-5.0f);
        max_point_z_m_ = scan_cfg["max_point_z_m"].as<float>(1.0f);

        const YAML::Node obs_cfg = env_cfg["observations"]["height_scan"]["params"];
        expected_scan_size_ = obs_cfg["size"].as<int>(0);

        size_x_cells_ = std::max(
            1, static_cast<int>(std::lround(size_x_m_ / resolution_m_)) + 1
        );
        size_y_cells_ = std::max(
            1, static_cast<int>(std::lround(size_y_m_ / resolution_m_)) + 1
        );
        const int computed_scan_size = size_x_cells_ * size_y_cells_;

        if (expected_scan_size_ != 0 && expected_scan_size_ != computed_scan_size) {
            spdlog::warn(
                "Height scan size mismatch: deploy.yaml expects {}, lidar config computes {}",
                expected_scan_size_,
                computed_scan_size
            );
        }

        pointcloud_sub_ =
            std::make_shared<unitree::robot::SubscriptionBase<PointCloudMsg>>(topic_);
        pointcloud_sub_->set_timeout_ms(pointcloud_timeout_ms_);

        {
            std::lock_guard<std::mutex> lock(robot_->data.height_scan_mutex);
            robot_->data.height_scan = empty_scan();
            robot_->data.height_scan_valid = false;
        }

        spdlog::info(
            "Configured lidar height scan on topic '{}' with grid {}x{} ({} values), mode='{}', debug_stats={}",
            topic_,
            size_x_cells_,
            size_y_cells_,
            computed_scan_size,
            height_scan_mode_name(height_scan_mode_),
            debug_stats_enabled_
        );
    }

    ~LidarHeightScanUpdater()
    {
        stop();
    }

    void start()
    {
        if (running_) {
            return;
        }
        running_ = true;
        worker_ = std::thread(&LidarHeightScanUpdater::run, this);
    }

    void stop()
    {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    std::vector<float> empty_scan() const
    {
        const int size = expected_scan_size_ > 0 ? expected_scan_size_ : size_x_cells_ * size_y_cells_;
        return std::vector<float>(size, miss_value_m_);
    }

    std::vector<float> apply_height_scan_mode(std::vector<float> scan) const
    {
        if (height_scan_mode_ == HeightScanMode::Zero) {
            std::fill(scan.begin(), scan.end(), 0.0f);
        } else if (height_scan_mode_ == HeightScanMode::MissValue) {
            std::fill(scan.begin(), scan.end(), miss_value_m_);
        }
        return scan;
    }

    void maybe_log_scan_stats(
        const std::vector<float> & raw_scan,
        bool valid,
        const size_t point_count
    )
    {
        if (!debug_stats_enabled_) {
            return;
        }

        using clock = std::chrono::steady_clock;
        const auto now = clock::now();
        if (last_stats_log_time_.time_since_epoch().count() != 0) {
            const auto elapsed_s = std::chrono::duration<double>(now - last_stats_log_time_).count();
            if (elapsed_s < std::max(debug_stats_period_s_, 0.1f)) {
                return;
            }
        }
        last_stats_log_time_ = now;

        size_t observed_cells = 0;
        float min_value = std::numeric_limits<float>::infinity();
        float max_value = -std::numeric_limits<float>::infinity();
        float sum_value = 0.0f;
        for (const float value : raw_scan) {
            if (!std::isfinite(value) || value >= miss_value_m_) {
                continue;
            }
            ++observed_cells;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum_value += value;
        }

        const float observed_ratio =
            raw_scan.empty() ? 0.0f : static_cast<float>(observed_cells) / static_cast<float>(raw_scan.size());

        if (observed_cells == 0) {
            spdlog::info(
                "[lidar] mode={} valid={} points={} observed_cells=0/{}, observed_ratio={:.3f}",
                height_scan_mode_name(height_scan_mode_),
                valid,
                point_count,
                raw_scan.size(),
                observed_ratio
            );
            return;
        }

        const float mean_value = sum_value / static_cast<float>(observed_cells);
        spdlog::info(
            "[lidar] mode={} valid={} points={} observed_cells={}/{}, observed_ratio={:.3f}, min={:.3f}, max={:.3f}, mean={:.3f}",
            height_scan_mode_name(height_scan_mode_),
            valid,
            point_count,
            observed_cells,
            raw_scan.size(),
            observed_ratio,
            min_value,
            max_value,
            mean_value
        );
    }

    void write_scan(std::vector<float> scan, bool valid)
    {
        scan = apply_height_scan_mode(std::move(scan));
        std::lock_guard<std::mutex> lock(robot_->data.height_scan_mutex);
        robot_->data.height_scan = std::move(scan);
        robot_->data.height_scan_valid = valid;
    }

    void run()
    {
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration<double>(
            1.0 / std::max(processing_rate_hz_, 1.0f)
        );
        auto next_tick = clock::now();

        while (running_) {
            update_once();
            next_tick += std::chrono::duration_cast<clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }
    }

    void update_once()
    {
        if (pointcloud_sub_->isTimeout()) {
            if (!timeout_warned_) {
                spdlog::warn("Waiting for lidar point cloud on '{}'", topic_);
                timeout_warned_ = true;
            }
            const std::vector<float> scan = empty_scan();
            maybe_log_scan_stats(scan, false, 0);
            write_scan(scan, false);
            return;
        }

        timeout_warned_ = false;

        PointCloudMsg msg;
        {
            std::lock_guard<std::mutex> lock(pointcloud_sub_->mutex_);
            msg = pointcloud_sub_->msg_;
        }

        if (msg.point_step() == 0 || msg.data().empty()) {
            const std::vector<float> scan = empty_scan();
            maybe_log_scan_stats(scan, false, 0);
            write_scan(scan, false);
            return;
        }

        const auto offsets = point_field_offsets(msg);
        if (!offsets.has_value()) {
            if (!field_warned_) {
                spdlog::warn(
                    "Point cloud '{}' is missing float32 x/y/z fields; skipping lidar update",
                    topic_
                );
                field_warned_ = true;
            }
            const std::vector<float> scan = empty_scan();
            maybe_log_scan_stats(scan, false, 0);
            write_scan(scan, false);
            return;
        }

        field_warned_ = false;

        std::vector<float> scan = empty_scan();
        std::vector<float> best_z(scan.size(), -std::numeric_limits<float>::infinity());

        const auto & point_bytes = msg.data();
        const size_t point_step = msg.point_step();
        const size_t point_count = point_bytes.size() / point_step;

        const Eigen::Quaternionf body_quat_w = robot_->data.root_quat_w;
        const Eigen::Quaternionf yaw_quat_w = isaaclab::yawQuaternion(body_quat_w);
        const Eigen::Quaternionf yaw_quat_inv = yaw_quat_w.conjugate();

        const float half_x = size_x_m_ * 0.5f;
        const float half_y = size_y_m_ * 0.5f;

        for (size_t point_idx = 0; point_idx < point_count; ++point_idx) {
            const size_t base = point_idx * point_step;
            const size_t x_addr = base + offsets->x;
            const size_t y_addr = base + offsets->y;
            const size_t z_addr = base + offsets->z;
            if (x_addr + sizeof(float) > point_bytes.size() ||
                y_addr + sizeof(float) > point_bytes.size() ||
                z_addr + sizeof(float) > point_bytes.size()) {
                continue;
            }

            const float x = read_float32(point_bytes, x_addr);
            const float y = read_float32(point_bytes, y_addr);
            const float z = read_float32(point_bytes, z_addr);

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }

            const Eigen::Vector3f point_lidar(x, y, z);
            const Eigen::Vector3f point_pelvis =
                lidar_rotation_in_pelvis_ * point_lidar + lidar_translation_in_pelvis_;
            const Eigen::Vector3f point_yaw =
                yaw_quat_inv * (body_quat_w * point_pelvis);

            if (point_yaw.x() < -half_x || point_yaw.x() > half_x ||
                point_yaw.y() < -half_y || point_yaw.y() > half_y ||
                point_yaw.z() < min_point_z_m_ || point_yaw.z() > max_point_z_m_) {
                continue;
            }

            int ix = static_cast<int>(std::floor((point_yaw.x() + half_x) / resolution_m_));
            int iy = static_cast<int>(std::floor((point_yaw.y() + half_y) / resolution_m_));
            ix = std::clamp(ix, 0, size_x_cells_ - 1);
            iy = std::clamp(iy, 0, size_y_cells_ - 1);
            const size_t scan_idx = static_cast<size_t>(iy * size_x_cells_ + ix);

            best_z[scan_idx] = std::max(best_z[scan_idx], point_yaw.z());
        }

        for (size_t i = 0; i < scan.size(); ++i) {
            if (std::isfinite(best_z[i])) {
                scan[i] = std::clamp(-best_z[i], 0.0f, miss_value_m_);
            }
        }

        maybe_log_scan_stats(scan, true, point_count);
        write_scan(std::move(scan), true);
    }

    isaaclab::Articulation * robot_;
    std::string topic_;
    float processing_rate_hz_ = 20.0f;
    int pointcloud_timeout_ms_ = 500;
    float size_x_m_ = 1.6f;
    float size_y_m_ = 1.0f;
    float resolution_m_ = 0.1f;
    float miss_value_m_ = 5.0f;
    float min_point_z_m_ = -5.0f;
    float max_point_z_m_ = 1.0f;
    int size_x_cells_ = 0;
    int size_y_cells_ = 0;
    int expected_scan_size_ = 0;
    Eigen::Vector3f lidar_translation_in_pelvis_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf lidar_rotation_in_pelvis_ = Eigen::Quaternionf::Identity();
    std::shared_ptr<unitree::robot::SubscriptionBase<PointCloudMsg>> pointcloud_sub_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    bool timeout_warned_ = false;
    bool field_warned_ = false;
    HeightScanMode height_scan_mode_ = HeightScanMode::Normal;
    bool debug_stats_enabled_ = false;
    float debug_stats_period_s_ = 1.0f;
    std::chrono::steady_clock::time_point last_stats_log_time_{};
};

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

}  // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    dry_run_no_actuation_ = cfg["dry_run_no_actuation"].as<bool>(false);
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const YAML::Node deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if (deploy_cfg["observations"]["height_scan"]) {
        lidar_updater_ = std::make_unique<LidarHeightScanUpdater>(
            env->robot.get(), cfg, deploy_cfg
        );
    }

    if (dry_run_no_actuation_) {
        spdlog::warn(
            "State '{}' is running in dry-run mode: lidar/policy active, motor actuation disabled",
            state_string
        );
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

State_RLBase::~State_RLBase()
{
    exit();
}

void State_RLBase::enter()
{
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[i];
        if (dry_run_no_actuation_) {
            motor.kp() = 0.0f;
            motor.kd() = 0.0f;
            motor.q() = lowstate->msg_.motor_state()[i].q();
        } else {
            motor.kp() = env->robot->data.joint_stiffness[i];
            motor.kd() = env->robot->data.joint_damping[i];
        }
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }

    env->robot->update();
    if (lidar_updater_) {
        lidar_updater_->start();
    }

    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desired_duration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);

        auto sleep_till = clock::now() + dt;
        env->reset();

        while (policy_thread_running)
        {
            env->step();
            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    });
}

void State_RLBase::run()
{
    if (dry_run_no_actuation_) {
        for (int i = 0; i < lowcmd->msg_.motor_cmd().size(); ++i) {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0.0f;
            motor.kd() = 0.0f;
            motor.q() = lowstate->msg_.motor_state()[i].q();
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
        }
        return;
    }

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_RLBase::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
    if (lidar_updater_) {
        lidar_updater_->stop();
    }
}
