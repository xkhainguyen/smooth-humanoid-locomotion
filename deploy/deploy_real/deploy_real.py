from typing import Union
import numpy as np
import time
import torch
import ipdb
import subprocess
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, quatToEuler
from common.remote_controller import RemoteController, KeyMap, KeyListener
from config import Config

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.key_listener = KeyListener()
 
        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        print("Loading jit for policy: ", config.policy_path)

        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tauj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_leg_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.update_mode_machine_ = False
        self.real_deploy = False


        self.cycle_time = 0.5
        self.n_priv = 0

        self.n_proprio = 2 + 3 + 3 + 2 + 2*self.config.num_actions + self.config.num_actions
        self.n_priv_latent = 4 + 1 + self.config.num_actions*2 + 3
            
        self.history_len = 10
        self.priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

            
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        print("Waiting for the robot to connect...")
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
            
    def LowStateHgHandler(self, msg: LowStateHG):
        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while not self.update_mode_machine_:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        # print("Waiting for the start signal...")

        # while (self.remote_controller.button[KeyMap.start] != 1):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def start_default_state(self):
        print("Moving to default state.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos state
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def keep_default_state(self):
        print("Enter default state.")
        # print("Waiting for the Button A signal...")

        # while self.remote_controller.button[KeyMap.A] != 1:
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def run(self):
        # OBSERVATION
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.action_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq
            self.tauj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].tau_est

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        waist_yaw = 0.0
        waist_yaw_omega = 0.0
    
        # Training uses old model, imu on torso, etc
        if self.config.imu_type == "pelvis":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)


        # create observation
        count = self.counter * self.config.control_dt
        phase = count % self.cycle_time / self.cycle_time
        sin_pos = [np.sin(2 * np.pi * phase)]
        cos_pos = [np.cos(2 * np.pi * phase)]

        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_21_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        rpy = quatToEuler(quat)
        rpy[1] += 0.098

        self.handle_key_cmd()

        obs_prop = np.concatenate([
            sin_pos, cos_pos,
            self.cmd,
            ang_vel,
            rpy[:2],
            qj_obs,
            dqj_obs,
            self.action,
        ])

        assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
        obs_hist = np.array(self.proprio_history_buf).flatten()
        self.proprio_history_buf.append(obs_prop)

        obs_buf = np.concatenate([obs_prop, self.priv_latent, obs_hist])
        assert obs_buf.shape[0] == self.config.num_obs, f"Expected {self.config.num_obs} but got {obs_buf.shape[0]}"
        
        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)

        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

        self.action = raw_action.copy()
        raw_action = np.clip(raw_action, -10., 10.)
        all_dof_actions = np.zeros(29)
        all_dof_actions[self.config.action_joint2motor_idx] = self.config.default_21_angles + raw_action * self.config.action_scale

        # transform action to self.target_leg_pos
        self.target_leg_pos = all_dof_actions[self.config.leg_joint2motor_idx]
        print("Target leg pos: ", self.target_leg_pos)

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_leg_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * 0.65
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * 1.1
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        target_upper_pos = all_dof_actions[self.config.arm_waist_joint2motor_idx] 
        # print(target_upper_pos)

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_upper_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] * 1.0
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] * 1.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt) # run every control_dt seconds

    def handle_key_cmd(self):
        # self.cmd[0] = self.remote_controller.ly
        # self.cmd[1] = self.remote_controller.lx * -1
        # # # self.cmd[0] = -1
        # self.cmd[2] = self.remote_controller.rx * -1
        
        # locomotion / leg movement in 4 directions and rotating
        if (self.key_listener.is_pressed('w')):
            self.cmd[0] += 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('s')):
            self.cmd[0] -= 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('a')):
            self.cmd[1] += 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('d')):
            self.cmd[1] -= 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('q')):
            self.cmd[2] += 0.1   
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('e')):
            self.cmd[2] -= 0.1
            print("[MODE] New command: ", self.cmd)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    config.default_21_angles = np.array([-0.4,  0.0,  0.0,  0.8, -0.35, 0.0,
                                         -0.4,  0.0,  0.0,  0.8, -0.35, 0.0,
                                         0,
                                         0., 0., 0, 0.,
                                         0., 0., 0, 0.])
    
    config.action_joint2motor_idx = np.array([0, 1, 2, 3, 4, 5, 
                                              6, 7, 8, 9, 10, 11,
                                              12, 
                                              15, 16, 17, 18, 
                                              22, 23, 24, 25])
    
    real_deploy = False if args.net == "lo" else True

    # Initialize DDS communication
    print("Initializing DDS communication...")
    channel = 1 if real_deploy is False else 0
    ChannelFactoryInitialize(channel, args.net)
    print("DDS communication initialized.")

    controller = Controller(config)

    if real_deploy:
        # Enter the zero torque state, press the start key to continue executing
        controller.zero_torque_state()

        # Gradually move to the default state
        controller.start_default_state()

        # Keep the robot at default state, press the A key to continue executing
        controller.keep_default_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
