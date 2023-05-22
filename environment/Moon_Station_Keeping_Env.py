#Environment for the RL station keeping problem for a spacecraft around
#the moon orbiting in a Halo orbit around L1

import gym
from gym import spaces
from pyrlprob.mdp import AbstractMDP

import numpy as np
from numpy.linalg import norm
from numpy import sqrt
from scipy.integrate import solve_ivp

from environment.equations import *
from environment.CR3BP import *
from environment.rk4 import *
from environment.BER4BP import *
from environment.pyKepler import *


class Moon_Station_Keeping_Env(AbstractMDP):

    def __init__(self, env_config):  #class constructor


        super().__init__(config=env_config)

        #observations: x, y, z, vx, vy, vz, m, t, dist_r, dist_v, theta, delta_C
        #actions: Fx, Fy, Fz, Fmod

        #Class attributes
        self.ueq=self.Isp*g0/v_star  #equivalent flux velocity
        self.time_step=self.tf/float(self.num_steps)  #time step
        self.dr_max /= l_star
        self.dv_max /= v_star

        self.num_obs=11  #number of observations 
        self.num_act=4  #number of actions
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,))
        self.action_space=spaces.Box(low=-1, high=1, shape=(self.num_act,))

        self.max_episode_steps=self.num_steps  #maximum number of episodes
        self.reward_range=(-float(np.inf),0)  #reward range (-inf,0)

        self.r_Halo_ref=0.83  #approximated r0 of the Halo (taken from the txt)
        self.v_Halo_ref=0.13  #approximated v0 of the Halo (taken from the txt)
        self.L1 = 0.8369  #L1 x position
        self.L2 = 1.1557  #L2 x position
        self.r_L1 = np.array([self.L1, 0, 0])
        self.r_L2 = np.array([self.L2, 0, 0])


        #Epsilon Law algorithm
        self.iter0 = self.eps_schedule[0][0]
        self.epsilon0 = self.eps_schedule[0][1]
        self.iterf = self.eps_schedule[1][0]
        self.epsilonf = self.eps_schedule[1][1]
        self.epsilon = self.epsilon0


        #Propagate the Halos
        self.num_steps_Halo = self.num_steps
        if self.dist_r_method == 3:
            self.num_steps_Halo = self.num_steps_Halo * 5

        self.r_Halo_all, self.v_Halo_all, self.T_Halo_all, self.C_Halo_all = data_Halos(self.filename, 2*self.num_steps_Halo, 2*self.tf, self.num_Halos)
        
        #print("r_Halo_all shape:" , np.shape(self.r_Halo_all))


    def get_observation(self, state, control):  #get the current observation

        step_Halo = state['step']
        if self.dist_r_method == 3:
            step_Halo = step_Halo*5

        r_obs = state['r'] - self.r_Halo[step_Halo]
        v_obs = state['v'] - self.v_Halo[step_Halo]
        m_obs = state['m']
        dist_r_obs = state['dist_r']
        dist_v_obs = state['dist_v']
        self.r_from_L1 = state['r'] - self.r_L1
        theta = arctan2(self.r_from_L1[1], self.r_from_L1[0])/(2*np.pi)  #angle between the x-axis and the relative sc position vector
        C_jacobi = Jacobi_constant(state['r'][0], state['r'][1], state['r'][2], state['v'][0], state['v'][1], state['v'][2])  
        delta_C = C_jacobi - self.C_Halo  #difference in Jacobi constant, additional observation

        observation=np.array([r_obs[0], r_obs[1], r_obs[2], v_obs[0], v_obs[1], \
                               v_obs[2], m_obs, dist_r_obs, dist_v_obs, theta, delta_C])

        return observation





    def get_control(self, action, state):  #get the control thrust F

        Fmod=0.5*(action[3]+1.)*self.Fmax
        F_vect=action[:3]
        F=Fmod*((F_vect)/norm(F_vect))

        return F




    def next_state(self, state, control, time_step):  #propagate the state


        if self.threebody:  #3 body dynamics
            s=np.array([state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], \
                       state['m']])  #current state

            t_span=[state['t'], state['t']+time_step]


            #Events
            hitMoon.terminal = True
            events = (hitMoon)

            data = np.concatenate((control, self.ueq), axis=None)

            #Solve equations of motion with CR3BP with control
            solution_int = solve_ivp(fun=CR3BP_equations_controlled_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(data,), rtol=1e-7, atol=1e-7)
            



        else:  #4 body dynamics
            s=np.array([state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], state['m'], state['anu_3']])  #current state

            t_span=[state['t'], state['t']+time_step]

            #Events
            hitMoon.terminal = True
            events = (hitMoon)

            data = np.concatenate((control, self.ueq, self.r0_sun, self.v0_sun, coe_moon ),axis=None)

            #Solve equations of motion with CR3BP
            solution_int = solve_ivp(fun=BER4BP_3dof, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(data,), rtol=1e-7, atol=1e-7)


        self.failure=solution_int.status

        r_new = np.array([solution_int.y[0][-1], solution_int.y[1][-1], solution_int.y[2][-1]])
        v_new = np.array([solution_int.y[3][-1], solution_int.y[4][-1], solution_int.y[5][-1]])
        m_new = solution_int.y[6][-1]

        state_next={'r':r_new, 'v':v_new,\
                    'm':m_new, 't':solution_int.t[-1], 'step':state['step']+1}  #next state

        if not self.threebody:
            state_next['anu_3'] = solution_int.y[7][-1]  #add moon anomaly if 4-body dynamics



        if self.dist_r_method==1:


            #Solve equations of motion with CR3BP
            t_span = [state_next['t'], state_next['t']+self.T_Halo]
            if self.threebody:
                s=np.concatenate((r_new, v_new), axis=None)
                data = []
                sol_afterperiod = solve_ivp(fun=CR3BP_equations_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events,
                                            args = (data,), \
                                            rtol=1e-7, atol=1e-7)
            else:
                s=np.array([r_new[0], r_new[1], r_new[2], \
                       v_new[0], v_new[1], v_new[2], m_new, state_next['anu_3']])  #current state
                data = np.concatenate((self.r0_sun, self.v0_sun, coe_moon ),axis=None)
                sol_afterperiod = solve_ivp(fun=BER4BP_3dof_free, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                    args=(data,), rtol=1e-7, atol=1e-7)

            r_afterperiod = np.array([sol_afterperiod.y[0][-1], sol_afterperiod.y[1][-1], sol_afterperiod.y[2][-1]])
            v_afterperiod = np.array([sol_afterperiod.y[3][-1], sol_afterperiod.y[4][-1], sol_afterperiod.y[5][-1]])

            delta_r=norm(r_afterperiod-r_new)  #error in distance with dist_r_method
            delta_v=norm(v_afterperiod-v_new)  #error in velocity with dist_r_method

        elif self.dist_r_method==2:
            
            data_sc = np.concatenate((r_new, v_new), axis=None)

            t_span = [state_next["t"], state_next["t"]+self.T_Halo]
            s_Halo = np.concatenate((self.r_Halo[state_next["step"]], self.v_Halo[state_next["step"]]), axis=None)

            min_dist_r.direction = 1
            min_dist_v.direction = 1
            events = (min_dist_r, min_dist_v)
            
            sol_Halo = solve_ivp(fun=CR3BP_equations_ivp, t_span=t_span, t_eval=None, y0=s_Halo, method='RK45', events=events,
                                            args = (data_sc,), \
                                            rtol=1e-7, atol=1e-7)
            
            dist_r_vect = [norm(r_new - self.r_Halo[state_next["step"]])]
            for i in range(len(sol_Halo.y_events[0])):
                r_i = sol_Halo.y_events[0][i][0:3]
                dist_r_i = norm(r_new - r_i) 
                dist_r_vect.append(dist_r_i)

            dist_v_vect = [norm(v_new - self.v_Halo[state_next["step"]])]
            for i in range(len(sol_Halo.y_events[1])):
                v_i = sol_Halo.y_events[1][i][3:6]
                dist_v_i = norm(v_new - v_i) 
                dist_v_vect.append(dist_v_i)

            delta_r = min(dist_r_vect)
            delta_v = min(dist_v_vect)

        elif self.dist_r_method == 3:

            dist_r_min = 1e10
            dist_v_min = 1e10
            for i in range(len(self.r_Halo)):
                dist_r_i = norm(r_new - self.r_Halo[i])
                dist_v_i = norm(v_new - self.v_Halo[i])
                if dist_r_i < dist_r_min:
                    dist_r_min = dist_r_i
                if dist_v_i < dist_v_min:
                    dist_v_min = dist_v_i
            
            delta_r = dist_r_min
            delta_v = dist_v_min

        else:
            delta_r=norm(r_new-self.r_Halo[state_next['step']])  #error in distance
            delta_v=norm(v_new-self.v_Halo[state_next['step']])  #error in velocity

        state_next['dist_r']=delta_r
        state_next['dist_v']=delta_v


        return state_next





    def collect_reward(self, prev_state, state, control):  #defines the reward function

        done=False  #episode not ended yet

        delta_r = state['dist_r']
        delta_v = state['dist_v']

        self.dist_r_mean=running_mean(self.dist_r_mean, state['step'], state['dist_r'])  #mean of dist_r
        self.dist_v_mean=running_mean(self.dist_v_mean, state['step'], state['dist_v'])  #mean of dist_v

        self.epsilon_r=self.epsilon*1e+03/l_star
        self.epsilon_v=self.epsilon*1/v_star
        delta_s=max(max((delta_r - self.epsilon_r)*self.w_r, delta_v - self.epsilon_v), 0)
        delta_m=prev_state['m']-state['m']

        reward = -(delta_s+self.w*delta_m)  #reward function definition

        if state['step']==self.num_steps:
            done=True
        if self.failure == 1:
            reward = reward - 10.*self.max_dist_r/l_star*(self.num_steps - state['step'])
            #reward = reward - 100*(self.num_steps - state['step'])
            done=True
        elif delta_r > self.max_dist_r/l_star:  #hard done if dist_r>max_dist_r
            reward = reward - 10.*self.max_dist_r/l_star*(self.num_steps - state['step'])
            done=True

        #reward = reward/10.

        return reward, done



    def get_info(self, prev_state, state, observation, control, reward, done):  #get current and final infos

        info={}
        info['episode_step_data']={}

        info['episode_step_data']['x']=[prev_state['r'][0]]
        info['episode_step_data']['y']=[prev_state['r'][1]]
        info['episode_step_data']['z']=[prev_state['r'][2]]
        info['episode_step_data']['vx']=[prev_state['v'][0]]
        info['episode_step_data']['vy']=[prev_state['v'][1]]
        info['episode_step_data']['vz']=[prev_state['v'][2]]
        info['episode_step_data']['m']=[prev_state['m']]
        info['episode_step_data']['t']=[prev_state['t']]
        info['episode_step_data']['Fx']=[control[0]]
        info['episode_step_data']['Fy']=[control[1]]
        info['episode_step_data']['Fz']=[control[2]]
        info['episode_step_data']['F_mod']=[norm(control)]
        info['episode_step_data']['dist_r']=[prev_state['dist_r']*l_star]
        info['episode_step_data']['dist_v']=[prev_state['dist_v']*v_star*1e3]

        if done:  #episode ended
            info['episode_end_data']={}
            info['custom_metrics']={}

            info['custom_metrics']['dist_r_mean']=self.dist_r_mean*l_star
            info['custom_metrics']['dist_v_mean']=self.dist_v_mean*v_star*1e3
            info['custom_metrics']['mf']=state['m']
            info['custom_metrics']['epsilon']=self.epsilon
            info['custom_metrics']['failure']=self.failure


            info['episode_step_data']['x'].append(state['r'][0])
            info['episode_step_data']['y'].append(state['r'][1])
            info['episode_step_data']['z'].append(state['r'][2])
            info['episode_step_data']['vx'].append(state['v'][0])
            info['episode_step_data']['vy'].append(state['v'][1])
            info['episode_step_data']['vz'].append(state['v'][2])
            info['episode_step_data']['m'].append(state['m'])
            info['episode_step_data']['t'].append(state['t'])
            info['episode_step_data']['Fx'].append(control[0])
            info['episode_step_data']['Fy'].append(control[1])
            info['episode_step_data']['Fz'].append(control[2])
            info['episode_step_data']['F_mod'].append(norm(control))
            info['episode_step_data']['dist_r'].append(state['dist_r']*l_star)
            info['episode_step_data']['dist_v'].append(state['dist_v']*v_star*1e3)


        return info




    def reset(self):  #reinitialize the process

        self.state={}
        self.state['step']=0
        self.state['t']=0.
        self.state['m']=self.m_sc
        control=0.

        if not self.threebody:  #4-body dynamics, we introduce the sun0
            anu_1_deg = np.random.uniform(0,360)  #sun true anomaly choosen randomly
            self.r0_sun, self.v0_sun = par2ic(coe_sun + [anu_1_deg*conv], sigma)  #trasform the classical orbital element set into position and velocity vectors
            anu_3 = np.random.uniform(0,360)  #moon true anomaly choosen randomly
            self.state['anu_3']=anu_3*conv

        # Choose the Halo
        k=randint(0,self.num_steps_Halo)  #
        i=randint(0,self.num_Halos-1)  #select a random matrix 

        self.r_Halo = self.r_Halo_all[i][k:k+self.num_steps_Halo+1]
        self.v_Halo = self.v_Halo_all[i][k:k+self.num_steps_Halo+1]
        self.T_Halo = self.T_Halo_all[i]
        self.C_Halo = self.C_Halo_all[i]

        self.r0 = self.r_Halo[0]
        self.v0 = self.v_Halo[0]

        # print("r_Halo shape:" , np.shape(self.r_Halo))
        # print("r_Halo vector:", self.r_Halo)
        # print("r0 vector:", self.r0)


        if self.error_initial_position:  #error on initial position and velocity (the error value is random but has a maximum)
            dr0 = np.random.uniform(-self.dr_max, self.dr_max, 3)  #initial position error vector
            dv0 = np.random.uniform(-self.dv_max, self.dv_max, 3)  #initial velocity error vector
            self.state['r'] = self.r0 + dr0
            self.state['v'] = self.v0 + dv0
            self.state['dist_r'] = norm(dr0)
            self.state['dist_v'] = norm(dv0)
            self.dist_r_mean = self.state['dist_r']
            self.dist_v_mean = self.state['dist_v']
        else:  #no error on initial position and velocity
            self.state['r'] = self.r0
            self.state['v'] = self.v0
            self.state['dist_r']=0.
            self.state['dist_v']=0.
            self.dist_r_mean=0.
            self.dist_v_mean=0.

       
        observation=self.get_observation(self.state, control)  #first observation to be returned as output

        return observation
