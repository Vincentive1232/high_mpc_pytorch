"""
A Simple Pendulum Gate

# ----------------------
# p = pivot point
# c = center
# 1,2,3,4 = corners
# ----------------------
#           p
#           |
#           |
#           |
#           |
#           |
#           |
#   2 - - - - - - - 1
#   |               |
#   |       c       |
#   |               |
#   4 - - - - - - - 3
#
"""

import numpy as np
from common.quadrotor_index import *
from common.pendulum_index import *
# from common.utils import Point

class Pendulum_v1(object):
    def __init__(self, pivot_point, sigma, T, dt):
        self.state_dim = 2
        self.action_dim = 0
        self.length = 2.0
        self.damping = 0.1
        self.mass = 2.0
        self.pi = 3.1415926
        self.gz = 9.81
        self.dt = dt
        self.pivot_point = pivot_point
        self.T = T

        self.sigma = sigma
        self.N = int(T/dt)
        self.width = 2.0
        self.height = 1


    def plan(self, state, opt_t=1.0):
        """
        根据给定的初始 state 状态，预测未来 N 个时间步的系统状态

        Args:
            :param state: current state of the pendulum
            :param opt_t： 一个优化时间目标（默认 1.0），用于输出中（不影响动力学计算）

        Return:
            plans:  包含预测的轨迹点（四元数表示 + 时间信息），变量 plans 是一个一维 Python 列表 (list)，
                    其结构是连续拼接的多个轨迹点数据。每个轨迹点是一个长度为7的列表，包含如下内容：
                    [traj_quat[0], traj_quat[1], traj_quat[2], traj_quat[3], current_t, opt_t, sigma]
                    plans的最终结构是一个长度为N*7的list
            pred_traj: 包含预测的轨迹点（欧拉角表示）
        """
        plans, pred_traj = [], []
        M = 4
        DT = self.dt/M   # 将dt拆分成四个更小的步长，从而提高rk4 integration的精度

        for i in range(self.N):
            x = state
            for _ in range(M):
                k1 = DT * self.f(x)
                k2 = DT * self.f(x + 0.5*k1)
                k3 = DT * self.f(x + 0.5*k2)
                k4 = DT * self.f(x + k3)

                x = x + (k1 + 2.0*(k2 + k3) + k4)/6.0

            state = x
            traj_euler_point = self.get_cartesian_state(state, euler=True).tolist()

            # plan trajectory and optimal time and optimal vx
            traj_quat_point = self.get_cartesian_state(state, euler=False).tolist()

            current_t = i * self.dt
            plan_i = traj_quat_point + [current_t, opt_t, self.sigma]      # sigma is the uncertainty of the prediction

            plans += plan_i
            pred_traj.append(traj_euler_point)

        return plans, pred_traj


    def f(self, state):
        theta = state[0]
        dot_theta = state[1]
        return np.array([dot_theta, -((self.gz/self.length)*np.sin(theta)+(self.damping/self.mass)*dot_theta)])


    def get_cartesian_state(self, state, euler=True):
        if not euler:
            cstate = np.zeros(shape=10)
            cstate[kPosX:kPosZ+1] = self.get_position(state)
            cstate[kQuatW:kQuatZ+1] = self.get_quaternion(state)
            cstate[kVelX:kVelZ+1] = self.get_velocity(state)
            return cstate
        else:
            cstate = np.zeros(shape=9)
            cstate[0:3] = self.get_position(state)
            cstate[3:6] = self.get_euler(state)
            cstate[6:9] = self.get_velocity(state)
            return cstate


    def get_position(self, state):
        """
        get position of the pendulum

        Args:
            state: the current state of the pendulum

        Return:
            np.array: position of the pendulum
        """
        pos = np.zeros(shape=3)
        pos[0] = self.pivot_point[0]
        pos[1] = self.pivot_point[1] + self.length * np.sin(state[kTheta])
        pos[2] = self.pivot_point[2] - self.length * np.cos(state[kTheta])
        return pos


    def get_velocity(self, state):
        """
        get linear velocity of the pendulum

        Args:

        Return:
            np.array: position of the pendulum
        """
        vel = np.zeros(shape=3)
        vel[0] = 0.0
        vel[1] = self.length * state[kDotTheta] * np.cos(state[kTheta])
        vel[2] = self.length * state[kDotTheta] * np.sin(state[kTheta])
        return vel

    @staticmethod
    def get_euler(state):
        """
        get euler angle of the pendulum

        Args:

        Return:
            np.array: euler angle of the pendulum
        """
        euler = np.zeros(shape=3)
        euler[0] = state[kTheta]
        euler[1] = 0.0
        euler[2] = 0.0
        return euler


    def get_quaternion(self, state):
        """
        transform euler angle to quaternion

        Args:

        Return:
            np.array: quaternion of the pendulum
        """
        roll, pitch, yaw = self.get_euler(state)

        cy = np.cos(yaw*0.5)
        sy = np.sin(yaw*0.5)
        cp = np.cos(pitch*0.5)
        sp = np.sin(pitch*0.5)
        cr = np.cos(roll*0.5)
        sr = np.sin(roll*0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        return [qw, qx, qy, qz]
