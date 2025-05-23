"""
High MPC which requires decision variables from a high-level policy
"""

import casadi as ca
import numpy as np
import time
from os import system

from common.quadrotor_index import *


class High_MPC(object):
    def __init__(self, T, dt, so_path='./nmpc.so'):
        """
        Nonlinear MPC for quadrotor control

        Args:
            :param T: Time horizon
            :param dt: Time step
            :param so_path: .so file path

        Return:
        """
        self.so_path = so_path

        # Time constant
        self.T = T
        self.dt = dt
        self.N = int(self.T / self.dt)

        # Gravity
        self.gz = 9.81

        # Quadrotor constant(for formalizing the constraints)
        self.w_max_yaw = 6.0
        self.w_max_xy = 6.0
        self.thrust_min = 2.0
        self.thrust_max = 20.0

        # state definition
        # [ px, py, pz,         # quadrotor position
        #   qw, qx, qy, qz,     # quadrotor quaternion
        #   vx, vy, vz ]        # quadrotor linear velocity
        self.state_dim = 10
        self.x = None

        # action definition
        # [ c_thrust, wx, wy, wz ]
        self.action_dim = 4
        self.u = None

        # constant cost matrix for tracking the goal point
        self.Q_goal = np.diag([
            100, 100, 100,   # delta_x, delta_y, delta_z
            10, 10, 10, 10,  # delta_qw, delta_qx, delta_qy, delta_qz
            10, 10, 10       # delta_vx, delta_vy, delta_vz
        ])

        # constant cost matrix for tracking the pendulum motion
        self.Q_pen = np.diag([
            0, 100, 100,     # delta_x, delta_y, delta_z
            10, 10, 10, 10,  # delta_qw, delta_qx, delta_qy, delta_qz
            0, 10, 10        # delta_vx, delta_vy, delta_vz
        ])

        # cost matrix for the action
        self.Q_u = np.diag([0.1, 0.1, 0.1, 0.1])  # T, wx, wy, wz

        # define initial state s0 and initial control input u0
        self.quad_s0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.quad_u0 = [9.81, 0.0, 0.0, 0.0]

        self.f = None

        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constraint functions, lbg < g
        self.ubg = []  # upper bound of constraint functions, g < ubg

        self.solver = None
        self.sol = None

        self.initDynamics()


    def initDynamics(self):
        """
        Define the control problem using casadi

        Args:

        Return:
        """
        # # # # # # # # # # # # # # # # # # # #
        # ------ Input States Definition ------
        # # # # # # # # # # # # # # # # # # # #
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        # concatenate the variables
        self.x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)



        # # # # # # # # # # # # # # # # # # # #
        # ----- Control Input Definition -----
        # # # # # # # # # # # # # # # # # # # #
        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        # concatenate the variables
        self.u = ca.vertcat(thrust, wx, wy, wz)



        # # # # # # # # # # # # # # # # # # # #
        # ---- System Dynamics Definition ----
        # # # # # # # # # # # # # # # # # # # #
        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * ( wx * qw + wz * qy - wy * qz),
            0.5 * ( wy * qw - wz * qx + wx * qz),
            0.5 * ( wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * thrust,
            2 * (qy * qz - qw * qx) * thrust,
            (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self.gz
        )

        self.f = ca.Function('f', [self.x, self.u], [x_dot], ['x', 'u'], ['ode'])

        F = self.sys_dynamics(self.dt)
        fMap = F.map(self.N, "openmp")  # parallel



        # # # # # # # # # # # # # # # # # # # #
        # ----- Loss Function Definition ------
        # # # # # # # # # # # # # # # # # # # #
        Delta_s = ca.SX.sym("Delta_s", self.state_dim)
        Delta_p = ca.SX.sym("Delta_p", self.state_dim)
        Delta_u = ca.SX.sym("Delta_u", self.action_dim)

        cost_goal = Delta_s.T @ self.Q_goal @ Delta_s
        cost_gap = Delta_p.T @ self.Q_pen @ Delta_p
        cost_u = Delta_u.T @ self.Q_u @ Delta_u

        # define a casadi function named after "cost_..."
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_gap = ca.Function('cost_gap', [Delta_p], [cost_gap])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])



        # # # # # # # # # # # # # # # # # # # #
        # ------ Nonlinear Optimization -------
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constraint functions, lbg < g
        self.ubg = []  # upper bound of constraint functions, g < ubg

        u_min = [self.thrust_min, -self.w_max_xy, -self.w_max_xy, -self.w_max_yaw]
        u_max = [self.thrust_max, self.w_max_xy, self.w_max_xy, self.w_max_yaw]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.state_dim)]
        x_max = [x_bound for _ in range(self.state_dim)]
        g_min = [0 for _ in range(self.state_dim)]
        g_max = [0 for _ in range(self.state_dim)]

        P = ca.SX.sym("P", self.state_dim + (self.state_dim + 3) * self.N + self.state_dim)  # define the reference point
        X = ca.SX.sym("X", self.state_dim, self.N + 1)  # define the state within the horizon as a matrix
        U = ca.SX.sym("U", self.action_dim, self.N)  # define the action

        X_next = fMap(X[:, :self.N], U)

        # "Lifted" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self.quad_s0
        self.lbw += x_min
        self.ubw += x_max

        # starting point
        self.nlp_g += [X[:, 0] - P[0:self.state_dim]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self.N):
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self.quad_u0
            self.lbw += u_min
            self.ubw += u_max

            # retrieve time constant
            idx_k = self.state_dim + self.state_dim + (self.state_dim + 3)*k
            idx_k_end = self.state_dim + (self.state_dim + 3)*(k+1)
            time_k = P[idx_k:idx_k_end]

            # # # # # # # # # # # # # # # # # # # #
            # ------ Compute exponential weights -------
            # - time_k[0] defines the current time
            # - time_k[1] defines the best traversal time, which is selected via
            #             a high-level policy / a deep high-level policy
            # - time_k[2] defines the temporal spread of the weight
            # # # # # # # # # # # # # # # # # # # #
            weight = ca.exp(-time_k[2] * (time_k[0] - time_k[1])**2)

            cost_goal_k = 0
            cost_gap_k = 0
            if k >= self.N - 1:
                # cost for tracking the goal position (这部分是goal cost)
                delta_s_k = (X[:, k + 1] - P[self.state_dim + (self.state_dim + 3) * self.N:])
                cost_goal_k = f_cost_goal(delta_s_k)
            else:
                # cost for tracking the moving gap (这部分是追踪dynamic gate的cost)
                delta_p_k = (X[:, k + 1] - P[self.state_dim + (self.state_dim + 3) * k:self.state_dim + (
                            self.state_dim + 3) * (k + 1) - 3])
                cost_gap_k = f_cost_gap(delta_p_k)

            # cost for tracking the given input
            delta_u_k = U[:, k] - [self.gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            # define the mpc objectives
            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_gap_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]  # 添加k+1步的状态变量
            self.nlp_w0 += self.quad_s0  # 添加该变量的初始猜测值
            self.lbw += x_min  # 添加该变量的下界
            self.ubw += x_max  # 添加该变量的上界

            # add equality constraints
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {'f': self.mpc_obj,
                    'x': ca.vertcat(*self.nlp_w),
                    'p': P,
                    'g': ca.vertcat(*self.nlp_g)}

        # Here 2 solver options are provided
        # # # # # # # # # # # # # # # # # # # #
        # ------------- qpoases ---------------
        # # # # # # # # # # # # # # # # # # # #
        '''
        nlp_options = {
            'verbose':False,
            'qpsol':"qpoases",
            "hessian_approximation": "gauss-newton",
            "max_iter":100,
            "tol_du":1e-2,
            "tol_pr":1e-2,
            "qpsol_options":{"sparse":True, "hessian_type":"posdef", "numRefinementSteps":1}
        }
        self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)      # 定义一个solver
        print("Generating shared library................")
        cname = self.solver.generate_dependencies("mpc_v1.c")                      # 生成c代码，用以加速
        system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)               # 用gcc进行编译，并且生成为一个.so文件
        self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)  # 用生成的c代码库重载solver，加速
        '''

        # # # # # # # # # # # # # # # # # # # #
        # -------------- ipopt ----------------
        # # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)         # 定义一个solver
        print("Generating shared library................")
        cname = self.solver.generate_dependencies("mpc_v1.c")                       # 生成c代码，用以加速
        system(f"gcc -shared -fPIC -o {self.so_path} {cname}")                      # 用gcc进行编译，并且生成为一个.so文件
        self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)  # 用生成的c代码库重载solver，加速


    def solve(self, ref_states):
        # # # # # # # # # # # # # # # # # # # #
        # ------------ Solve NLP --------------
        # # # # # # # # # # # # # # # # # # # #
        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=ref_states,
            lbg=self.lbg,
            ubg=self.ubg
        )
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self.state_dim:self.state_dim+self.action_dim]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self.state_dim+self.action_dim:2*(self.state_dim+self.action_dim)]) + list(sol_x0[self.state_dim+self.action_dim:])

        x0_array = np.reshape(sol_x0[:-self.state_dim], newshape=(-1, self.state_dim+self.action_dim))

        return opt_u, x0_array


    def sys_dynamics(self, dt):
        M = 4
        DT = dt/M
        X0 = ca.SX.sym("X", self.state_dim)
        U = ca.SX.sym("U", self.action_dim)

        X = X0
        for _ in range(M):
            k1 = DT*self.f(X, U)
            k2 = DT*self.f(X + 0.5*k1, U)
            k3 = DT*self.f(X + 0.5*k2, U)
            k4 = DT*self.f(X + k3, U)

            X = X + (k1 + 2*(k2 + k3) + k4)/6

        F = ca.Function('F', [X0, U], [X])
        return F