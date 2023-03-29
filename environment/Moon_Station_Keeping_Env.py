#Environment for the RL station keeping problem for a spacecraft around 
#the moon orbiting in a Halo orbit around L1

import gym
from gym import spaces
from pyrlprob.mdp import AbstractMDP

import numpy as np
from numpy.linalg import norm
from numpy import sqrt

from environment.equations import *
from environment.constants import *
from environment.LOR_RL_cislunar.CR3BP import *


class Moon_Station_Keeping_Env(AbstractMDP):

    def __init__(self, env_config):  #class constructor
        
        
        super().__init__(config=env_config)

        #observations: x, y, z, vx, vy, vz, m, t, dist_r, dist_v
        #actions: Fx, Fy, Fz, Fmod

        #obtain randomly the L1 Halo initial conditions
        r0,v0=choose_Halo('/home/carlo/RL_cislunar/Halo_L1_rv', single_matrix=False)

        #class attributes
        num_steps=100.  #number of time intervals
        self.ueq=self.Isp*g0/v_star  #equivalent flux velocity
        self.dt=self.tf/float(self.num_steps)  #time step
       
        self.num_obs=10.  #number of observations
        self.num_act=4.  #number of actions
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,))
        self.action_space=spaces.Box(low=-1, high=1, shape=(self.num_act,))
        
        self.max_episode_steps=self.num_steps  #maximum number of episodes
        self.reward_range=(-float(np.inf),0)  #reward range


        

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
        



    def get_control(self, state, action):  #get the control thrust F

        Fmod=0.5*(action[-1]+1)*self.Fmax
        F_vect=action[:2]
        F=Fmod*((F_vect)/norm(F_vect))

        return F
    



    def next_state(self, state, control, dt):  #propagate the state

        state=np.array(state['r'][0], state['r'][1], state['r'][2], \
                       state['v'][0], state['v'][1], state['v'][2], \
                       state['m'], state['t'], state['dist_r'], state['dist_v'])  #current state 

        num_steps=100.  #number of time intervals
        t_span=[state['t'], state['t']+dt]  #time interval 
        t_eval=np.linspace(t_span[0], t_span[1], num_steps)  #time vector

        data=np.array([control, self.ueq])  #integration data
        solution_int=rk4(CR3BP, state, t_eval, data)  #obtain the solution by integration (CR3BP)
        
        self.success=True
        state_next={'x':solution_int[0], 'y':solution_int[1], 'z':solution_int[2], \
                    'vx':solution_int[3], 'vy':solution_int[4], 'vz':solution_int[5], \
                    'm':solution_int[6], 't':solution_int[7], 'step':state['step']+1}  #next state
        
        return state_next
    


    

    def collect_reward(self, state_prev, state, control):  #defines the reward function
        
        done=False  #episode not ended yet
        
        x=state[0]
        y=state[1]
        z=state[2]
        r=sqrt(x**2+y**2+z**2)

        vx=state[3]
        vy=state[4]
        vz=state[5]
        v=sqrt(vx**2+vy**2+vz**2)

        delta_r=norm(r-self.r_Halo[state['step']])  #error in distance
        delta_v=norm(v-self.v_Halo)  #error in velocity

        state['dist_r']=delta_r
        state['dist_v']=delta_v
        self.dist_r_mean=self.running_mean(self.dist_r_mean, state['step'], state['dist_r'])
        self.dist_v_mean=self.running_mean(self.dist_v_mean, state['step'], state['dist_v'])

        reward=-delta_r-delta_v  #reward function definition (to be maximized!)
        
        if state['step']==self.num_steps:
            done=True
        if self.success is False:
            done=True
        
        return reward, done



    def running_mean(self, mean, step, new_value):  #calculate the mean of the state at every step
        
        new_mean=1./float(step)*((float(step)-1.)*mean+new_value)

        return new_mean



    def get_info(self, state_prev, state, observation, control, reward, done):  #get current and final infos

        info={}
        info['episode_step_data']={}

        if done is False:  #episode not ended yet

            #definisci con r --> r[0], r[1 etc??? 
            info['episode_step_data']['r']=[state_prev['r']]
            info['episode_step_data']['v']=[state_prev['v']]

            info['episode_step_data']['x']=[state_prev['r'][0]]
            info['episode_step_data']['y']=[state_prev['r'][1]]
            info['episode_step_data']['z']=[state_prev['r'][2]]
            info['episode_step_data']['vx']=[state_prev['v'][0]]
            info['episode_step_data']['vy']=[state_prev['v'][1]]
            info['episode_step_data']['vz']=[state_prev['v'][2]]
            info['episode_step_data']['m']=[state_prev['m']]
            info['episode_step_data']['t']=[state_prev['t']]
            info['episode_step_data']['dist_r']=[state_prev['dist_r']]
            info['episode_step_data']['dist_v']=[state_prev['dist_v']]

        else:  #episode ended
            info['episode_end_data']={}
            info['custom_metrics']={}

            info['custom_metrics']['dist_r']=state['dist_r']
            info['custom_metrics']['dist_v']=state['dist_v']

            #snellire anche qui???
            info['episode_step_data']['x'].append(state['x'])
            info['episode_step_data']['y'].append(state['y'])
            info['episode_step_data']['z'].append(state['z'])
            info['episode_step_data']['vx'].append(state['vx'])
            info['episode_step_data']['vy'].append(state['vy'])
            info['episode_step_data']['vz'].append(state['vz'])
            info['episode_step_data']['m'].append(state['m'])
            info['episode_step_data']['t'].append(state['t'])
            info['episode_step_data']['dist_r'].append(state['dist_r'])
            info['episode_step_data']['dist_v'].append(state['dist_v'])
            

        return info




    def reset(self):  #reinitialize the process

        self.state={}
        self.state['step']=0.
        self.state['t']=0.
        self.state['m']=self.m_sc
        self.state['dist_r']=0.
        self.state['dist_v']=0.
        self.dist_r_mean=0.
        self.dist_v_mean=0.
        control=0.
        
        self.r0, self.v0= choose_Halo(self.filename, self.single_matrix)
        self.state['r']=self.r0

        observation=self.get_observation(self.state, control)  #first observation to be returned as output

        return observation




r_Halo, v_Halo = 