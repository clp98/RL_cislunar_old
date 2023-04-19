#Environment for the RL station keeping problem for a spacecraft around 
#the moon orbiting in a Halo orbit around L2

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


class Moon_Station_Keeping_Env(AbstractMDP):

    def __init__(self, env_config):  #class constructor
        
        
        super().__init__(config=env_config)

        #observations: x, y, z, vx, vy, vz, m, t, dist_r, dist_v
        #actions: Fx, Fy, Fz, Fmod


        #class attributes
        self.ueq=self.Isp*g0/v_star  #equivalent flux velocity
        self.time_step=self.tf/float(self.num_steps)  #time step time_step
       
        self.num_obs=10  #number of observations
        self.num_act=4  #number of actions
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,))
        self.action_space=spaces.Box(low=-1, high=1, shape=(self.num_act,))
        
        self.max_episode_steps=self.num_steps  #maximum number of episodes
        self.reward_range=(-float(np.inf),0)  #reward range (-inf,0)

        self.r_Halo_ref=0.83  #approximated r0 of the Halo (taken from the txt)
        self.v_Halo_ref=0.13  #approximated v0 of the Halo (taken from the txt)

        #Epsilon Law
        self.iter0 = self.eps_schedule[0][0]
        self.epsilon0 = self.eps_schedule[0][1]
        self.iterf = self.eps_schedule[1][0]
        self.epsilonf = self.eps_schedule[1][1]
        self.epsilon = self.epsilon0


        

    def get_observation(self, state, control):  #get the current observation

        r_obs=state['r']
        v_obs=state['v']
        m_obs=state['m']
        t_obs=state['t']
        dist_r_obs=state['dist_r']
        dist_v_obs=state['dist_v']


        observation=np.array([r_obs[0], r_obs[1], r_obs[2], v_obs[0], v_obs[1], \
                               v_obs[2], m_obs, t_obs, dist_r_obs, dist_v_obs])

        return observation
        



    def get_control(self, action, state):  #get the control thrust F

        Fmod=0.5*(action[3]+1.)*self.Fmax
        F_vect=action[:3]
        F=Fmod*((F_vect)/norm(F_vect))

        return F
    



    def next_state(self, state, control, time_step):  #propagate the state 



        '''if CR3BP:  #CR3BP equations of motion
            s=np.array([state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], \
                       state['m']])  #current state 

            t_span=[state['t'], state['t']+time_step]  #time interval 

            #Events
            hitMoon.terminal = True
            events = (hitMoon)

            #Solve equations of motion with CR3BP
            solution_int = solve_ivp(fun=CR3BP_equations_controlled_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(control, self.ueq), rtol=1e-7, atol=1e-7)
            
        else:  #BER4BP equations of motion
            s=np.array([state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], state['m'], anu_3])  #current state 
            
            t_span=[state['t'], state['t']+time_step]

            #Events
            hitMoon.terminal = True
            events = (hitMoon)

            #Solve equations of motion with CR3BP
            solution_int = solve_ivp(fun=BER4BP_3dof, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(control, self.ueq), rtol=1e-7, atol=1e-7)'''
            

        s=np.array([state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], \
                       state['m']])  #current state 

        t_span=[state['t'], state['t']+time_step]  #time interval 

        #Events
        hitMoon.terminal = True
        events = (hitMoon)

        #Solve equations of motion with CR3BP
        solution_int = solve_ivp(fun=CR3BP_equations_controlled_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
            args=(control, self.ueq), rtol=1e-7, atol=1e-7)


        self.success=True

        r_new = np.array([solution_int.y[0][-1], solution_int.y[1][-1], solution_int.y[2][-1]])
        v_new = np.array([solution_int.y[3][-1], solution_int.y[4][-1], solution_int.y[5][-1]])
        m_new = solution_int.y[6][-1]

        state_next={'r':r_new, 'v':v_new,\
                    'm':m_new, 't':solution_int.t[-1], 'step':state['step']+1}  #next state
    
        
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

        delta_s=max(max(delta_r/self.r_Halo_ref, delta_v/self.v_Halo_ref)-self.epsilon, 0)
        delta_m=prev_state['m']-state['m']

        reward=-(delta_s+self.w*delta_m)  #reward function definition
        
        #reward update RIFARE!!! CON AUTOVETTORE USCENTE
        #lambda_pos=2.15 #autovalore positivo divergente
        #reward=-(delta_s*lambda_pos+self.w*delta_m)

        if state['step']==self.num_steps:
            done=True
        if self.success is False:
            done=True
        
        reward /= 100.
        
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
        info['episode_step_data']['dist_r']=[prev_state['dist_r']]
        info['episode_step_data']['dist_v']=[prev_state['dist_v']]

        if done:  #episode ended
            info['episode_end_data']={}
            info['custom_metrics']={}

            info['custom_metrics']['dist_r']=self.dist_r_mean
            info['custom_metrics']['dist_v']=self.dist_v_mean
            info['custom_metrics']['mf']=state['m']

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
            info['episode_step_data']['dist_r'].append(state['dist_r'])
            info['episode_step_data']['dist_v'].append(state['dist_v'])
            

        return info




    def reset(self):  #reinitialize the process

        self.state={}
        self.state['step']=0
        self.state['t']=0.
        self.state['m']=self.m_sc
        self.state['dist_r']=0.
        self.state['dist_v']=0.
        self.dist_r_mean=0.
        self.dist_v_mean=0.
        control=0.
        
        self.r0, self.v0=choose_Halo(self.filename, self.single_matrix)

        if self.error_initial_position:  #error on initial position and velocity is present (the error value is random)
            dr0 = np.random.rand(3)  #initial position error vector
            dv0 = np.random.rand(3)  #initial velocity error vector
            norm_dr0 = np.linalg.norm(dr0)
            norm_dv0 = np.linalg.norm(dv0)
            if norm_dr0/self.r0 < 0.01:
                self.state['r'] = self.r0 + dr0
            if norm_dv0/self.v0 < 0.01:
                self.state['v'] = self.v0 + dv0
        else:  
            self.state['r'] = self.r0
            self.state['v'] = self.v0

        #self.state['r'] = self.r0
        #self.state['v'] = self.v0

        self.r_Halo,self.v_Halo=rv_Halo(self.r0, self.v0, 0, self.tf, self.num_steps)  #recall rv_Halo function to obtain reference Halo position and velocity

        observation=self.get_observation(self.state, control)  #first observation to be returned as output

        return observation

