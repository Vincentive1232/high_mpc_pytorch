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
from common.pendulum_index import *
from common.utils import Point

class Pendulum_v0(object):
    def __init__(self, pivot_point, dt):
        self.state_dim = 2
        self.action_dim = 0

        # define some physical constants
        self.damping = 0.1
        self.mass = 2.0
        self.gz = 9.81
        self.dt = dt
        self.pivot_point = pivot_point

        # define the state variable of the pendulum
        self.state = np.zeros(shape=self.state_dim)

        # define initial state
        self.theta_box = np.array([-0.8, 0.8]) * np.pi
        self.dot_theta_box = np.array([-0.1, 0.1]) * np.pi

        # x, y, z, roll, pitch, yaw, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])

        # define some constants of the pendulum
        self.length = 2.0          # distance between pivot point to the gate center
        self.width = 1.0           # gate width  (for visualization only)
        self.height = 0.5          # gate height (for visualization only)
        self.length1 = 0.0         # distance between pivot point to corner 1
        self.length2 = 0.0         # distance between pivot point to corner 2
        self.length3 = 0.0         # distance between pivot point to corner 3
        self.length4 = 0.0         # distance between pivot point to corner 4
        self.delta_theta1 = 0.0    # angle between pivot point to corner 1
        self.delta_theta2 = 0.0    # angle between pivot point to corner 2
        self.delta_theta3 = 0.0    # angle between pivot point to corner 3
        self.delta_theta4 = 0.0    # angle between pivot point to corner 4


        # initialize the pendulum
        self.t = 0.0
        self.init_corners()
        self.reset()


    def init_corners(self):
        # compute distance between pivot point to 4 corners and 4 angles
        edge1, edge2 = self.width/2, self.length - self.height/2
        self.length1 = np.sqrt(edge1**2 + edge2**2)
        self.delta_theta1 = np.arctan2(edge1, edge2)
        self.length2 = self.length1
        self.delta_theta2 = -self.delta_theta1

        edge1, edge2 = self.width / 2, self.length + self.height / 2
        self.length3 = np.sqrt(edge1 ** 2 + edge2 ** 2)
        self.delta_theta3 = np.arctan2(edge1, edge2)
        self.length4 = self.length3
        self.delta_theta4 = -self.delta_theta3


    def reset(self, init_theta=None):
        if init_theta is not None:
             # 如果有预设的初始状态则直接设置为单摆的初始位置
            self.state[kTheta] = init_theta
        else:
            # 否则在一个有限区间内进行采样得到初始状态
            self.state[kTheta] = np.random.uniform(
                low=self.theta_box[0],
                high=self.theta_box[1]
            )
        self.state[kDotTheta] = np.random.uniform(
            low=self.dot_theta_box[0],
            high=self.dot_theta_box[1]
        )

        self.t = 0.0
        print("init pendulum: ", self.state)
        return self.state


    def rk4_integration(self):
        self.t = self.t + self.dt

        # RK4 integration
        M = 4
        DT = self.dt/M

        X = self.state
        for _ in range(M):
            k1 = DT*self.f(X)
            k2 = DT*self.f(X + 0.5 * k1)
            k3 = DT*self.f(X + 0.5 * k2)
            k4 = DT*self.f(X + k3)

            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0

        self.state = X
        return self.state


    def f(self, state):
        theta = state[0]
        dot_theta = state[1]
        return np.array([dot_theta, -((self.gz/self.length)*np.sin(theta)+(self.damping/self.mass)*dot_theta)])


    def get_state(self):
        return self.state


    @property
    def get_t(self):
        return self.t


    def get_cartesian_state(self):
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()

        return cartesian_state


    def get_position(self):
        pos = np.zeros(shape=3)
        pos[0] = self.pivot_point[0]
        pos[1:] = self.to_planar_coordinates(self.pivot_point, l=self.length, theta=self.state[kTheta])

        return pos


    def get_velocity(self):
        vel = np.zeros(shape=3)
        vel[0] = 0.0
        vel[1] = self.length*self.state[kDotTheta]*np.cos(self.state[kTheta])   # linear velocity = l * angular_velocity
        vel[2] = self.length*self.state[kDotTheta]*np.sin(self.state[kTheta])   # linear velocity = l * angular_velocity

        return vel


    def get_euler(self):
        euler = np.zeros(shape=3)
        euler[0] = self.state[kTheta]
        euler[1] = 0.0
        euler[2] = 0.0

        return euler


    @staticmethod
    def to_planar_coordinates(pivot_point, l, theta):
        # 将gate的中心点坐标转换到平面坐标系上
        y = pivot_point[1] + l*np.sin(theta)
        z = pivot_point[2] - l*np.cos(theta)

        return y, z


    def get_corners(self):
        # 得到pendulum各个角点的2D位置，基本是为了visualization
        theta = self.state[kTheta]
        y1, z1 = self.to_planar_coordinates(self.pivot_point, self.length1, theta + self.delta_theta1)
        y2, z2 = self.to_planar_coordinates(self.pivot_point, self.length2, theta + self.delta_theta2)
        y3, z3 = self.to_planar_coordinates(self.pivot_point, self.length3, theta + self.delta_theta3)
        y4, z4 = self.to_planar_coordinates(self.pivot_point, self.length4, theta + self.delta_theta4)

        corners = [Point(x=y1, y=z1), Point(x=y2, y=z2), Point(x=y3, y=z3), Point(x=y4, y=z4)]

        return corners


    def get_3d_corners(self):
        # 得到pendulum各个角点的3D位置，基本是为了visualization
        theta = self.state[kTheta]
        y1, z1 = self.to_planar_coordinates(self.pivot_point, self.length1, theta + self.delta_theta1)
        y2, z2 = self.to_planar_coordinates(self.pivot_point, self.length2, theta + self.delta_theta2)
        y3, z3 = self.to_planar_coordinates(self.pivot_point, self.length3, theta + self.delta_theta3)
        y4, z4 = self.to_planar_coordinates(self.pivot_point, self.length4, theta + self.delta_theta4)

        x = self.pivot_point[0]
        corners_3d = [[x, y1, z1], [x, y2, z2], [x, y3, z3], [x, y4, z4]]

        return corners_3d



