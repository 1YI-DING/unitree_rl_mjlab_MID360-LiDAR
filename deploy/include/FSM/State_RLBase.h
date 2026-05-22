// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class LidarHeightScanUpdater;

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    ~State_RLBase();
    
    void enter();

    void run();
    
    void exit();

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<LidarHeightScanUpdater> lidar_updater_;
    bool dry_run_no_actuation_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
