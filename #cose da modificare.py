#cose da modificare



#reward update
lambda_pos=2.15 #autovalore positivo divergente (preso da appunti Circi)
reward=-(delta_s*lambda_pos+self.w*delta_m)



#error on initial position and velocity


if error_initial_position:
    dr0=10. #error on initial position [km]
    dv0=0.2 #error on initial velocity [km/s]
    self.r_Halo,self.v_Halo=rv_Halo(self.r0+dr0, self.v0+dv0, 0, self.tf, self.num_steps)
else:
    self.r_Halo,self.v_Halo=rv_Halo(self.r0, self.v0, 0, self.tf, self.num_steps)


