import numpy as np
from scipy.spatial.transform import Rotation as R
from common.quadrotor_index import *

class Quadrotor_v0(object):
    def __init__(self, dt):
        self.state_dim = 10
        self.action_dim = 4

        self._state = np.zeros(shape=self.state_dim)
        self._state[kQuatW] = 1.0

        self._actions = np.zeros(shape=self.action_dim)

        self._gz = 9.81
        self._dt = dt
        self._arm_1 = 0.3   # m, the length of the quadrotor arm

        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array([
            [-3.0, -1.0],  # x
            [-2.0,  2.0],  # y
            [ 0.0,  2.5]   # z
        ])
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array([
            [-1.0,  1.0],  # x
            [-1.0,  1.0],  # y
            [-1.0,  1.0]   # z
        ])

        # obstacles position upper bound and lower bound
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])

        # reset when initialized
        self.reset()


    def reset(self):
        """
        initialize / reset the state of the quadrotor

        Args:

        Return:
            np.array: The initialized state of the quadrotor
        """
        self._state = np.zeros(shape=self.state_dim)
        self._state[kQuatW] = 1.0

        # randomly initialize position
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1]
        )
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1]
        )
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1]
        )

        # randomly initialize rotation
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        self._state[kQuatW:kQuatZ+1] = quad_quat0 / np.linalg.norm(quad_quat0)   # normalize it to be unit quaternion

        # randomly initialize velocity
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1]
        )
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1]
        )
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1]
        )

        return self._state


    def rk4_propagation(self, action):
        """
        propagate the quadrotor state using RK4 integration

        Args:
            action: the control input of the quadrotor dynamics

        Return:
            np.array: The current state of the quadrotor
        """
        M = 4
        DT = self._dt / M

        X = self._state
        for i in range(M):
            k1 = DT*self.f(X, action)
            k2 = DT*self.f(X + 0.5*k1, action)
            k3 = DT*self.f(X + 0.5*k2, action)
            k4 = DT*self.f(X + k3, action)

            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0

        self._state = X
        return self._state


    def f(self, state, action):
        """
        quadrotor dynamics: ds = f(x, u)

        Args:
            state: current state before propagation
            action: the control input of the quadrotor dynamics

        Return:
            np.array: The current state of the quadrotor
        """
        thrust, wx, wy, wz = action
        dstate = np.zeros(shape=self.state_dim)

        dstate[kPosX:kPosZ+1] = state[kVelX:kVelZ+1]

        qw, qx, qy, qz = self.get_quaternion()

        dstate[kQuatW] = 0.5 * (-wx*qx - wy*qy - wz*qz)
        dstate[kQuatX] = 0.5 * ( wx*qw + wz*qy - wy*qz)
        dstate[kQuatY] = 0.5 * ( wy*qw - wz*qx + wx*qz)
        dstate[kQuatZ] = 0.5 * ( wz*qw + wy*qx - wx*qy)

        dstate[kVelX] = 2 * (qw*qy + qx*qz) * thrust
        dstate[kVelY] = 2 * (qy*qz - qw*qx) * thrust
        dstate[kVelZ] = (qw*qw - qx*qx - qy*qy + qz*qz) * thrust - self._gz

        return dstate


    def set_state(self, state):
        """
        set internal state

        Args:
            state: the state to be set

        Return:
        """
        self._state = state


    def get_state(self):
        """
        get internal state

        Args:

        Return:
            np.array: current state of the quadrotor
        """
        return self._state


    def get_cartesian_state(self):
        """
        transform the current state into cartesian coordinates

        Args:

        Return:
            np.array: cartesian state
        """
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        return cartesian_state


    def get_position(self):
        """
        get position of the quadrotor

        Args:

        Return:
            np.array: position of the quadrotor
        """
        return self._state[kPosX:kPosZ+1]


    def get_velocity(self):
        """
        get linear velocity of the quadrotor

        Args:

        Return:
            np.array: position of the quadrotor
        """
        return self._state[kVelX:kVelZ+1]


    def get_quaternion(self):
        """
        Retrieve quaternion

        Args:

        Return:
            np.array: quaternion of the quadrotor
        """
        quat = self._state[kQuatW:kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        return quat


    def get_euler(self):
        """
        Retrieve euler angles of the quadrotor

        Args:

        Return:
            np.array: position of the quadrotor
        """
        quat = self.get_quaternion()
        euler = self.quatToEuler(quat)
        return euler


    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)

        Args:

        Return:
            list: 3 axes in world frame
        """
        b_x = np.array([self._arm_1, 0, 0])
        b_y = np.array([0, self._arm_1, 0])
        b_z = np.array([0, 0, -self._arm_1])

        quat = self.get_quaternion()
        rot_matrix = R.from_quat(quat=quat).as_matrix()
        quad_center = self.get_position()

        w_x = rot_matrix @ b_x + quad_center
        w_y = rot_matrix @ b_y + quad_center
        w_z = rot_matrix @ b_z + quad_center
        return [w_x, w_y, w_z]


    def get_motor_pos(self):
        """
        Get 4 motor poses in world frame (for visualization only)

        Args:

        Return:
            list: 4 motor poses in world frame
        """
        b_motor1 = np.array([ np.sqrt(self._arm_1/2),  np.sqrt(self._arm_1/2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_1/2),  np.sqrt(self._arm_1/2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_1/2), -np.sqrt(self._arm_1/2), 0])
        b_motor4 = np.array([ np.sqrt(self._arm_1/2), -np.sqrt(self._arm_1/2), 0])

        quat = self.get_quaternion()
        rot_matrix = R.from_quat(quat=quat).as_matrix()
        quad_center = self.get_position()

        # transform it into world frame
        w_motor1 = rot_matrix @ b_motor1 + quad_center
        w_motor2 = rot_matrix @ b_motor2 + quad_center
        w_motor3 = rot_matrix @ b_motor3 + quad_center
        w_motor4 = rot_matrix @ b_motor4 + quad_center

        return [w_motor1, w_motor2, w_motor3, w_motor4]


    @staticmethod
    def quatToEuler(quat):
        """
        Convert Quaternion to Euler Angles

        Args:

        Return:
            list: euler angle of the quadrotor
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2 * quat_w * quat_x + 2 * quat_y * quat_z,
                             quat_w * quat_w - quat_x * quat_x - quat_y * quat_y + quat_z * quat_z)
        euler_y = -np.arcsin(2 * quat_x * quat_z - 2 * quat_w * quat_y)
        euler_z = np.arctan2(2 * quat_w * quat_z + 2 * quat_x * quat_y,
                             quat_w * quat_w + quat_x * quat_x - quat_y * quat_y - quat_z * quat_z)
        return [euler_x, euler_y, euler_z]
