import gym
from gym import spaces

import scipy
import numpy as np
from numpy.linalg import norm
from numpy.random import uniform
from numpy.random import normal
from numpy.random import randint
from sklearn.neighbors import KDTree

from pyrlprob.mdp import AbstractMDP

from environment.LOR_RL_cislunar.CR3BP import *
from environment.LOR_RL_cislunar.min_dist import *


""" RL CISLUNAR ENVIRONMENT CLASS """
class CislunarEnv(AbstractMDP):
    """
    Reinforcement Learning environment,
    for a minimum-propellant, time-free, low-thrust transfer
    between two Libration point orbits.
    The spacecraft trajectory is divided in NSTEPS segments, and 
    the CR3BP equations of motion, with low-thrust terms, are used to propagate the spacecraft state
    between any two time-steps. The (nondimensioanl) state components are defined with respect 
    to the barycenter of a Earth-Moon rotating reference frame.  
    The environment can be deterministic or stochastic 
    (i.e., with dynamical and/or navigation uncertainties).
    The reference initial state of the chaser can be randomly sampled along the initial orbit, or
    can be fixed. Moreover, it can be randomly perturbed.

    Class inputs:
        - env_config:
            (bool) planar:        0 = 3D problem
                                  1 = planar problem
            (int) action_type:   0 = action in cartesian coordinates
                                 1 = action in cartesian coordinates + t_1 + t_2
            (bool) randomIC:    True = random initial conditions,
                                False = fixed initial conditions
            (bool) pertIC:      True = perturbed initial conditions,
                                False = nominal initial conditions
            (bool) nav_errors:  True = navigation errors,
                                False = no navigation errors
            (bool) fixed_point_f:   True = fixed final point
                                    False = free final point                
            (int) NSTEPS: number of trajectory segments
            (list) eps_schedule: list of tolerance values for 
                    terminal constraint satisfaction on terminal state
            (float) eta: weight of terminal constraint violation in reward
            (float) fmax: maximum engine thrust, non-dim
            (float) Isp: specific impulse, s
            (float) tf: maximum mission time, non-dim
            (float) m0: initial spacecraft mass, non-dim
            (list) r0_in: initial position on departure orbit, non-dim
            (list) v0_in: initial velocity on departure orbit, non-dim
            (float) T_in: period of the departure orbit, non-dim
            (list) r0_f: initial condition defining the arrival orbit (position), non-dim
            (list) v0_f: initial condition defining the arrival orbit (velocity), non-dim
            (float) T_f: period of the arrival orbit, non-dim
            (list) dr0_max: maximum variation of initial position, non-dim
            (list) dv0_max: maximum variation of initial velocity, non-dim
            (list) sigma_r: standard deviation on position, non-dim
            (list) sigma_v: standard deviation on velocity, non-dim
            (float) sigma_m: standard deviation on mass, non-dim


            
    RL ENVIRONMENT (MDP)
    Observations: 
        Type: Box(9)
        Num	Observation                Min     Max
        0	x                        -max_r   max_r
        1	y                        -max_r   max_r
        2	z                        -max_r   max_r
        3	vx                       -max_v   max_v
        4	vy                       -max_v   max_v
        5	vz                       -max_v   max_v
        6	m                           0       1
        7   t                           0       tf
        8   dmin                        0      inf
        
    Actions:
    if action_type == 0:
        Type: Box(4)
        Num	Action
        0	fnorm                      -1       1
        1	fx                         -1       1
        2	fy                         -1       1
        3	fz                         -1       1
    elif action_type == 1:
        Type: Box(6)
        Num	Action
        0	fnorm                      -1       1
        1	fx                         -1       1
        2	fy                         -1       1
        3	fz                         -1       1
        4   eta_1                       0       1
        5   eta_2                       0       1

    Starting State:
        Start at state: (r0, v0)
    
    Episode Termination:
        - When the target orbit is reached
        - At time tf
        - Position and/or velocity reach the environment boundaries

    """


    def __init__(self, env_config):
        """ 
        Class constructor 
        """

        super().__init__(config=env_config)

        """ Class attributes """
        self.r0_in = np.array(self.r0_in)
        self.v0_in = np.array(self.v0_in)
        self.r0_f = np.array(self.r0_f)
        self.v0_f = np.array(self.v0_f)
        self.dr0_max = np.array(self.dr0_max)
        self.dv0_max = np.array(self.dv0_max)
        self.sigma_r = np.array(self.sigma_r)
        self.sigma_v = np.array(self.sigma_v)
        self.ueq = self.Isp*g0/v_star
        
        """ Time variables """
        self.time_step = self.tf / float(self.H)     # time step, non-dim

        """ Environment boundaries """
        self.max_r = 0.5                     # maximum distance from the Moon, nondim
        self.min_r = 0.005                   # minimum distance from the Moon, nondim
        self.max_v = 2                       # maximum velocity, nondim

        """ Moon and L1/L2 positions """
        self.L1 = 0.8369
        self.L2 = 1.1557
        self.rMoon = np.array([1 - mu, 0., 0.]) #nondim

        """ States along the initial and target orbit """
        self.s0_in = np.concatenate((self.r0_in, self.v0_in), axis = None)
        self.s0_f = np.concatenate((self.r0_f, self.v0_f), axis = None)
        self.Npoints = 2500
        t_eval_in = np.linspace(0., self.T_in, self.Npoints)
        t_eval_f = np.linspace(0., self.T_f, self.Npoints)
        self.sorb_in = \
            propagate_cr3bp_free(self.s0_in, t_eval = t_eval_in)
        self.sorb_f = \
            propagate_cr3bp_free(self.s0_f, t_eval = t_eval_f)

        """ Final point """ 
        if self.fixed_point_f:  #True --> fixed final point
            self.state_f = self.sorb_f[int(len(self.sorb_f)/2 - 1)]
        else:  #False --> free final point
            self.tree = KDTree(self.sorb_f, metric='euclidean')
        
        """ Epsilon law """  #(see pyrlprob documentation)
        self.iter0 = self.eps_schedule[0][0]
        self.epsilon0 = self.eps_schedule[0][1]
        self.iterf = self.eps_schedule[1][0]
        self.epsilonf = self.eps_schedule[1][1]
        self.epsilon = self.epsilon0

        """ OBSERVATION/ACTION SPACE """
        if self.planar: #planar problem
            self.n_obs = 7
            self.n_act = 3
        else: #3D problem
            self.n_obs = 9
            self.n_act = 4
        if self.action_type: #Box(6) ergo add +2 components
            self.n_act = self.n_act + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,))
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.n_act,))  #action acts in the (-1,+1) range

        self.max_episode_steps = self.H
        self.reward_range = (-float('inf'), 0.) 


    def collect_reward(self, prev_state, state, control):
        """
        Collect Reward
        """

        done = False

        # Frequent reward: none
        reward = state['m'] - prev_state['m']

        # Episode end
        if state["event"] or state["step"] == self.H:

            done = True
            
            # Constraint violation
            c_viol_r, c_viol_v, c_viol_s = \
                self.cstr_violation(state['dist'], state['rNN'], state['vNN'], \
                    state['rCP'], state['vCP'])
            c_viol = max(0., c_viol_s - self.epsilon)
            state["c_viol_r"] = c_viol_r
            state["c_viol_v"] = c_viol_v
            state["c_viol_s"] = c_viol_s

            reward = reward - self.eta*c_viol #update the reward function
                
        return reward, done
    
    ###

    def cstr_violation(self, dist, rNN, vNN, r, v):
        """
        Constraint violation on closest state
        """

        c_viol_r = norm(r - rNN)
        c_viol_v = norm(v - vNN)
        c_viol_s = dist

        return c_viol_r, c_viol_v, c_viol_s
    

    def NearestNeighbor(self, state):
        """
        Nearest neighbor on the target orbit.

        Args:
            state: spacecraft state (r,v) along the current trajectory
        
        Return:
            rNN: position of the closest nearest neighbor along the target orbit to state
            vNN: velocity of the closest nearest neighbor along the target orbit to state
            dist: distance between state and nearest neighbor
        """

        if not self.fixed_point_f: #free final point
            dist, ind = self.tree.query([state], k=1)

            stateNN = self.sorb_f[ind[0][0]]
            rNN = stateNN[0:3]
            vNN = stateNN[3:6]
            dist = dist[0][0]
        else: #fixed final point
            rNN = self.state_f[0:3]
            vNN = self.state_f[3:6]
            dist = norm(state - self.state_f)

        return rNN, vNN, dist


    def IC_variation(self):
        """
        Random variation of the initial conditions

        Returns:
            dr0 (np.array): variation of the initial position
            dv0 (np.array): variation of the initial velocity

        """

        #Position and velocity error
        dx0 = float(uniform(-self.dr0_max[0], self.dr0_max[0]))
        dy0 = float(uniform(-self.dr0_max[1], self.dr0_max[1]))
        dz0 = float(uniform(-self.dr0_max[2], self.dr0_max[2]))
        dr0 = np.array([dx0, dy0, dz0])

        #Velocity error
        dvx0 = float(uniform(-self.dv0_max[0], self.dv0_max[0]))
        dvy0 = float(uniform(-self.dv0_max[1], self.dv0_max[1]))
        dvz0 = float(uniform(-self.dv0_max[2], self.dv0_max[2]))
        dv0 = np.array([dvx0, dvy0, dvz0])

        return dr0, dv0


    def state_uncty(self):
        """
        State uncertainty

        Return:
            dr (np.array): uncertainty on position
            dv (np.array): uncertainty on velocity
            dm (np.array): uncertainty on mass
        """

        #Position error
        dx = float(normal(0., self.sigma_r[0], 1))
        dy = float(normal(0., self.sigma_r[1], 1))
        dz = float(normal(0., self.sigma_r[2], 1))
        dr = np.array([dx, dy, dz])

        #Velocity error
        dvx = float(normal(0., self.sigma_v[0], 1))
        dvy = float(normal(0., self.sigma_v[1], 1))
        dvz = float(normal(0., self.sigma_v[2], 1))
        dv = np.array([dvx, dvy, dvz])

        #mass error
        dm = float(normal(0., self.sigma_m, 1))

        return dr, dv, dm
    
    
    def get_observation(self, state, control=None):
        """
        Get current observation
        """

        # Get navigation errors
        if self.nav_errors: #with navigation errors
            dr, dv, dm = self.state_uncty()
            r_obs = state["r"] + dr
            v_obs = state["v"] + dv
            m_obs = state["m"] + dm
        else: #without navigation errors
            r_obs = state["r"]
            v_obs = state["v"]
            m_obs = state["m"]

        # Observations
        if self.planar: #planar problem (7 components)
            obs = np.array([r_obs[0], r_obs[1], \
                v_obs[0], v_obs[1], \
                m_obs, state["t"], state["dist"]])
        else: #3D problem (9 components)
            obs = np.array([r_obs[0], r_obs[1], r_obs[2], \
                v_obs[0], v_obs[1], v_obs[2], \
                m_obs, state["t"], state["dist"]])
        
        return obs


    def get_control(self, action, state):
        """
        Get current control
        """

        fnorm = 0.5*(action[0]+1)*self.fmax

        if self.action_type:
            fdir = action[1:-2]
        else:
            fdir = action[1:]
        if self.planar:
            fdir = np.append(fdir, 0.) 
        
        f = fnorm*fdir/norm(fdir)

        if self.action_type:
            eta_1 = 0.5*(action[-2]+1)*self.time_step
            eta_2 = 0.5*(action[-1]+1)*(self.time_step - eta_1)
            self.t_1 = state['t'] + eta_1
            self.t_2 = self.t_1 + eta_2
        else:
            self.t_1 = state['t']
            self.t_2 = state['t'] + self.time_step
        
        return f
    

    def min_dist_from_target_orbit(self, t, y, f, t_1, t_2, ueq):

        setattr(CislunarEnv.min_dist_from_target_orbit, "terminal", True)

        #Current state
        state = y[:6]

        #Nearest neighbor to current state
        _, _, dist = self.NearestNeighbor(state)

        return dist - self.epsilon
    

    def next_state(self, state, control, time_step):
        """
        Propagate state
        """

        #State before propagation
        s = np.array([state['r'][0], state['r'][1], state['r'][2], \
           state['v'][0], state['v'][1], state['v'][2], state['m']])

        #State at the next time step
        N_points_step = 50
        t_span = [state['t'], state['t'] + time_step]
        t_eval = np.linspace(t_span[0], t_span[1], N_points_step)

        #Events
        hitMoon.terminal = True
        events = (hitMoon, self.min_dist_from_target_orbit)

        #Solve equations of motion
        sol = solve_ivp(fun=CR3BP_eqs, t_span=t_span, t_eval=t_eval, y0=s, method='RK45', events=events, \
            args=(control, self.t_1, self.t_2, self.ueq), rtol=1e-7, atol=1e-7)

        r_new = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
        v_new = np.array([sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
        m_new = sol.y[6][-1]

        #Dense solution
        self.t_dense = sol.t
        self.r_dense = sol.y[0:3]
        self.v_dense = sol.y[3:6]
        self.m_dense = sol.y[6]
        self.f_dense = [list(control) if t >= self.t_1 and t <= self.t_2 else [0., 0., 0.] for t in self.t_dense]
        self.f_norm_dense = [norm(f) for f in self.f_dense]
        self.f_dense = np.transpose(self.f_dense)

        t_final = sol.t[-1]
        if sol.status == 1:
            event_triggered = True
        else:
            event_triggered = False
        
        #Closest state to target orbit
        min_dist = state["dist"]
        rNN = state["rNN"]
        vNN = state["vNN"]
        rCP = state["rCP"]
        vCP = state["vCP"]
        for j in range(len(self.t_dense)):
            state_dense = np.concatenate((np.array([self.r_dense[0][j], self.r_dense[1][j], self.r_dense[2][j]]), \
                np.array([self.v_dense[0][j], self.v_dense[1][j], self.v_dense[2][j]])), axis = None)
            rNN_dense, vNN_dense, dist = self.NearestNeighbor(state_dense)
            if dist < min_dist:
                min_dist = dist
                rNN = rNN_dense
                vNN = vNN_dense
                rCP = state_dense[0:3]
                vCP = state_dense[3:6]

        #State at next time-step
        s_new = {"r": r_new, "v": v_new, "m": m_new, "t": t_final, \
            "dist": min_dist, "rNN": rNN, "vNN": vNN, "rCP": rCP, "vCP": vCP, \
            "step": state["step"] + 1, "event": event_triggered}
        
        return s_new


    def get_info(self, prev_state, state, observation, control, reward, done):
        """
        Get current info
        """

        #Set info
        info = {}
        info["episode_step_data"] = {}
        if not done:
            info["episode_step_data"]['t'] = self.t_dense[:-1]
            info["episode_step_data"]['x'] = self.r_dense[0][:-1]
            info["episode_step_data"]['y'] = self.r_dense[1][:-1]
            info["episode_step_data"]['z'] = self.r_dense[2][:-1]
            info["episode_step_data"]['vx'] = self.v_dense[0][:-1]
            info["episode_step_data"]['vy'] = self.v_dense[1][:-1]
            info["episode_step_data"]['vz'] = self.v_dense[2][:-1]
            info["episode_step_data"]['m'] = self.m_dense[:-1]
            info["episode_step_data"]['fx'] = self.f_dense[0][:-1]
            info["episode_step_data"]['fy'] = self.f_dense[1][:-1]
            info["episode_step_data"]['fz'] = self.f_dense[2][:-1]
            info["episode_step_data"]['fnorm'] = self.f_norm_dense[:-1]
        else:
            info["episode_end_data"] = {}
            info["custom_metrics"] = {}
            info["episode_step_data"]['t'] = self.t_dense
            info["episode_step_data"]['x'] = self.r_dense[0]
            info["episode_step_data"]['y'] = self.r_dense[1]
            info["episode_step_data"]['z'] = self.r_dense[2]
            info["episode_step_data"]['vx'] = self.v_dense[0]
            info["episode_step_data"]['vy'] = self.v_dense[1]
            info["episode_step_data"]['vz'] = self.v_dense[2]
            info["episode_step_data"]['m'] = self.m_dense
            info["episode_step_data"]['fx'] = self.f_dense[0]
            info["episode_step_data"]['fy'] = self.f_dense[1]
            info["episode_step_data"]['fz'] = self.f_dense[2]
            info["episode_step_data"]['fnorm'] = self.f_norm_dense
            info["episode_end_data"]['x0'] = self.r0[0]
            info["episode_end_data"]['y0'] = self.r0[1]
            info["episode_end_data"]['z0'] = self.r0[2]
            info["episode_end_data"]['vx0'] = self.v0[0]
            info["episode_end_data"]['vy0'] = self.v0[1]
            info["episode_end_data"]['vz0'] = self.v0[2]
            info["episode_end_data"]['xf'] = state['r'][0]
            info["episode_end_data"]['yf'] = state['r'][1]
            info["episode_end_data"]['zf'] = state['r'][2]
            info["episode_end_data"]['vxf'] = state['v'][0]
            info["episode_end_data"]['vyf'] = state['v'][1]
            info["episode_end_data"]['vzf'] = state['v'][2]
            info["episode_end_data"]['tf'] = state['t']
            info["episode_end_data"]['xCP'] = state['rCP'][0]
            info["episode_end_data"]['yCP'] = state['rCP'][1]
            info["episode_end_data"]['zCP'] = state['rCP'][2]
            info["episode_end_data"]['vxCP'] = state['vCP'][0]
            info["episode_end_data"]['vyCP'] = state['vCP'][1]
            info["episode_end_data"]['vzCP'] = state['vCP'][2]
            info["episode_end_data"]['xNN'] = state['rNN'][0]
            info["episode_end_data"]['yNN'] = state['rNN'][1]
            info["episode_end_data"]['zNN'] = state['rNN'][2]
            info["episode_end_data"]['vxNN'] = state['vNN'][0]
            info["episode_end_data"]['vyNN'] = state['vNN'][1]
            info["episode_end_data"]['vzNN'] = state['vNN'][2]
            info["episode_end_data"]['dist'] = state['dist']
            info["custom_metrics"]['c_viol_r'] = state['c_viol_r']
            info["custom_metrics"]['c_viol_v'] = state['c_viol_v']
            info["custom_metrics"]['c_viol_s'] = state['c_viol_s']
            info["custom_metrics"]['mf'] = state['m']
            info["custom_metrics"]['epsilon'] = self.epsilon
        
        return info


    """ Initialize the episode """
    def reset(self):
        """
        return obs: observation vector

        """

        # Environment variables
        if self.randomIC:
            rand_ind = np.random.choice(self.Npoints)
            self.r0 = self.rorb_in[rand_ind]    # position at the k-th time step, non-dim
            self.v0 = self.vorb_in[rand_ind]    # velocity at the k-th time step, non-dim
        else:
            self.r0 = self.r0_in                        # position at the k-th time step, non-dim
            self.v0 = self.v0_in                        # velocity at the k-th time step, non-dim
        
        # Random initial conditions
        if self.pertIC:
            dr0, dv0 = self.IC_variation()
            self.r0 = self.r0 + dr0
            self.v0 = self.v0 + dv0

        #Initial distance 
        state0 = np.concatenate((self.r0, self.v0), axis = None)
        rNN, vNN, dist = self.NearestNeighbor(state0)

        #Initial state
        self.state = {}
        self.state['r'] = self.r0
        self.state['v'] = self.v0    
        self.state['m'] = self.m0    
        self.state['t'] = 0.
        self.state['dist'] = dist   
        self.state['rNN'] = rNN 
        self.state['vNN'] = vNN 
        self.state['rCP'] = self.r0 
        self.state['vCP'] = self.v0 
        self.state['step'] = 0       
        self.state['event'] = False

        # Observations
        obs = self.get_observation(self.state)

        return obs