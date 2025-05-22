import numpy as np

from simulation.quadrotor import Quadrotor_v0
from simulation.pendulum_v0 import Pendulum_v0
from simulation.pendulum_v1 import Pendulum_v1

from common.quadrotor_index import *


# # # # # # # # # # # # # # # # # # # #
# --------- Define a Sampler ----------
# # # # # # # # # # # # # # # # # # # #
class Space(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)


# # # # # # # # # # # # # # # # # # # #
# ----------- Dynamic Gap -------------
# # # # # # # # # # # # # # # # # # # #
class Dynamic_Gap(object):
    def __init__(self, mpc, plan_T, plan_dt):
        self.mpc = mpc
        self.plan_T = plan_T
        self.plan_dt = plan_dt

        self.goal_point = np.array([4.0, 0.0, 2.0])
        self.pivot_point = np.array([2.0, 0.0, 3.0])

        # define goal state, position, quaternion, velocity
        self.quad_sT = self.goal_point.tolist() + [1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]

        # Simulation Parameters
        self.sim_T = 3.0
        self.sim_dt = 0.02
        self.max_episode_steps = int(self.sim_T/self.sim_dt)

        # instantiate class objects: a quadrotor and a pendulum
        self.quad = Quadrotor_v0(dt=self.sim_dt)
        self.pend = Pendulum_v0(self.pivot_point, dt=self.sim_dt)

        # instantiate a planner
        self.planner = Pendulum_v1(self.pivot_point, sigma=10, T=self.plan_T, dt=self.plan_dt)

        # define a constraint observation space for sampling
        self.observation_space = Space(
            low=np.array([-10.0, -10.0, -10.0, -2*np.pi, -2*np.pi, -2*np.pi, -10.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0, 2*np.pi, 2*np.pi, 2*np.pi, 10.0, 10.0, 10.0])
        )

        # define a constraint action space for sampling
        self.action_space = Space(
            low=np.array([0.0]),
            high=np.array([2*self.plan_T])
        )

        # define the state of the quadrotor and the pendulum
        self.quad_state = None
        self.pend_state = None

        # reset the environment
        self.t = 0
        self.reset()


    @staticmethod
    def seed(seed):
        np.random.seed(seed=seed)


    def reset(self, init_theta=None):
        self.t = 0

        # state for ODE
        self.quad_state = self.quad.reset()
        if init_theta is not None:
            self.pend_state = self.pend.reset(init_theta)
        else:
            self.pend_state = self.pend.reset()

        # observation, which can be part of the state, e.g., position
        # or a cartesian representation of the state
        quad_obs = self.quad.get_cartesian_state()
        pend_obs = self.pend.get_cartesian_state()

        obs = (quad_obs - pend_obs).tolist()

        return obs


    def step(self, u=0):
        self.t += self.sim_dt
        opt_t = u

        # generate planned trajectory
        plan_pend_traj, pred_pend_traj_cart = self.planner.plan(self.pend_state, opt_t)
        pred_pend_traj_cart = np.array(pred_pend_traj_cart)

        quad_s0 = self.quad_state.tolist()
        ref_traj = quad_s0 + plan_pend_traj + self.quad_sT      # 拼接初始状态，预测途中轨迹(门的轨迹)，终点状态，形成完整参考轨迹

        # run nonlinear model predictive control
        quad_act, pred_traj = self.mpc.solver(ref_traj)

        # run the actual control command on the quadrotor
        self.quad_state = self.quad.rk4_propagation(quad_act)
        # simulate one step on the pendulum
        self.pend_state = self.pend.rk4_integration()

        # update the observation
        quad_obs = self.quad.get_cartesian_state()
        pend_obs = self.pend.get_cartesian_state()
        obs = (quad_obs - pend_obs).tolist()

        # define the info
        info = {
            "quad_obs": quad_obs,
            "quad_act": quad_act,
            "quad_axes": self.quad.get_axes(),
            "pend_obs": pend_obs,
            "pend_corners": self.pend.get_3d_corners(),
            "pred_quad_traj": pred_traj,
            "pred_pend_traj": pred_pend_traj_cart,
            "opt_t": opt_t,
            "plan_dt": self.plan_dt
        }
        done = False
        if self.t >= (self.sim_T - self.sim_dt):
            done = True

        return obs, 0, done, info


    def episode(self, u):
        opt_t = u

        plan_pend_traj, pred_pend_traj_cart = self.planner.plan(self.pend_state, opt_t)
        pred_pend_traj_cart = np.array(pred_pend_traj_cart)

        quad_s0 = self.quad_state.tolist()
        ref_traj = quad_s0 + plan_pend_traj + self.quad_sT

        _, pred_traj = self.mpc.solve(ref_traj)

        # 保证索引 opt_node 范围合法：最小为 0，最大不超过 pred_traj 的最后一个索引。
        opt_node = np.clip(int(opt_t/self.plan_dt), 0, pred_traj.shape[0]-1)
        opt_min = np.clip(opt_node - 10, 0, pred_traj.shape[0] - 1)     # opt_min 是往前回溯 10 步，但不能小于 0
        opt_max = np.clip(opt_node + 5,  0, pred_traj.shape[0] - 1)     # opt_max 是向前看 5 步，但不能超过数组最大索引

        loss = np.linalg.norm(pred_traj[opt_min:opt_max, kPosX:kPosZ+1] - pred_pend_traj_cart[opt_min:opt_max, kPosX:kPosZ+1])
        reward = -loss

        return reward, opt_node


    @staticmethod
    def is_within_gap(gap_corners, point):
        """
        judge whether a point is in a 2d polygon or not(using half space test)

        Args:
            :param gap_corners: the corner points of the polygon
            :param point： the point to be judged

        Return:

        """
        A, B, C = [], [], []
        for i in range(len(gap_corners)):
            p1 = gap_corners[i]
            p2 = gap_corners[(i + 1) % len(gap_corners)]

            # calculate A, B, C
            a = -(p2.y - p1.y)
            b = p2.x - p1.x
            c = -(a * p1.x + b * p1.y)

            A.append(a)
            B.append(b)
            C.append(c)

        D = []
        for i in range(len(A)):
            d = A[i] * point.x + B[i] * point.y + C[i]
            D.append(d)

        t1 = all(d >= 0 for d in D)
        t2 = all(d <= 0 for d in D)

        return t1 or t2
