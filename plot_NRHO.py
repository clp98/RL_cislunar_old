# import numpy as np
# import astropy.coordinates as coord
# import astropy.units as u
# import matplotlib.pyplot as plt

# # Define the initial positions and velocities

# x_Halo_L1=[]
# y_Halo_L1=[]
# z_Halo_L1=[]
# vx_Halo_L1=[]
# vy_Halo_L1=[]
# vz_Halo_L1=[]
# T_Halo_L1=[]

# with open('PO_files/NRHO_L2_rv.txt', "r") as file_L1: 
#     file_L1.readline()
#     file_all=file_L1.readlines()
#     for line in file_all: #read line
#         line=line.split()
#         state=np.array(line).astype(np.float64) #non-dimensional data
        
#         #save data
#         x_Halo_L1.append(state[0])
#         y_Halo_L1.append(state[1])
#         z_Halo_L1.append(state[2])
#         vx_Halo_L1.append(state[3])
#         vy_Halo_L1.append(state[4])
#         vz_Halo_L1.append(state[5])
#         T_Halo_L1.append(state[6])

# file_L1.close()

# positions =  [x_Halo_L1, y_Halo_L1, z_Halo_L1] # List of initial positions (x, y, z)
# velocities = [vx_Halo_L1, vy_Halo_L1, vz_Halo_L1]  # List of initial velocities (vx, vy, vz)

# # Create a Heliocentric frame
# heliocentric = coord.ICRS(x=positions[:, 0] * u.pc, y=positions[:, 1] * u.pc, z=positions[:, 2] * u.pc,
#                           v_x=velocities[:, 0] * u.km/u.s, v_y=velocities[:, 1] * u.km/u.s,
#                           v_z=velocities[:, 2] * u.km/u.s)

# # Transform to Galactocentric frame
# galactocentric = heliocentric.transform_to(coord.Galactocentric)

# # Extract the transformed positions
# x = galactocentric.x.value
# y = galactocentric.y.value
# z = galactocentric.z.value

# # Plot the orbits
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('X [pc]')
# ax.set_ylabel('Y [pc]')
# ax.set_zlabel('Z [pc]')
# plt.show()
