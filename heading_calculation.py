# Function to take in Iridium STL observables and generate a heading in the NED frame
# Landon Boyd
# 06/10/2024

import numpy as np
import math
import navtools.conversions as conversions
import pymap3d as pm

def unit_vectors(rx_pos:np.ndarray,sat_pos:np.ndarray):
    # Function to extract the unit vectors and geometric range estimates for GPS positioning
    # Landon Boyd
    
    sat_rel_pos = sat_pos - rx_pos
    ranges = np.linalg.norm(sat_rel_pos,ord=2,axis=1)
    unit_vectors = sat_rel_pos / ranges[:,None]
    return unit_vectors,ranges

# def leo_pdr_heading(lla_0:np.array,
#                     measured_doppler:np.array,
#                     current_pos:np.array,
#                     current_vel:np.array,
#                     sat_pos:np.array,
#                     sat_vel:np.array):
    
#     # 0 vertical pos and vel
#     current_ecef = conversions.enu2ecef(current_pos[0],current_pos[1],0,lla_0[0],lla_0[1],lla_0[2],deg=False)
#     current_ecef_vel = conversions.enu2uvw(current_vel[0],current_vel[1],0,lla_0[0],lla_0[1],deg=False)
    
#     # Unit Vectors and gemoetry, we'll see how the clock bias assumption goes
#     u_vecs,_ = unit_vectors(current_ecef,sat_pos)

#     # Estimated dopplers are velocity difference projected onto unit vectors
#     sat_rel_vel = sat_vel - current_ecef_vel
#     dopp_hat = []

#     num_sats = np.ma.size(u_vecs,0)
#     for count in range(num_sats):
#         dopp_hat.append(np.dot(u_vecs[count,:],sat_rel_vel[count,:]))
        
#     y_hat = np.array(dopp_hat)

#     # Estimate user velocities
#     del_y = measured_doppler - y_hat
#     vel_est = np.linalg.pinv(u_vecs)@del_y

#     # Rotate back into enu
#     # vel_est_enu = conversions.ecef2enu(vel_est[0],vel_est[1],vel_est[2],lla_0[0],lla_0[1],lla_0[2],deg=False)
#     vel_est_enu = pm.ecef2enuv(vel_est[0],vel_est[1],vel_est[2],lla_0[0],lla_0[1],deg=False)

#     # Estimate heading
#     heading = math.atan2(vel_est_enu[0],vel_est_enu[1])

#     return heading


def leo_pdr_heading_enu(lla_0:np.array,
                        measured_doppler:np.array,
                        current_pos:np.array,
                        current_vel:np.array,
                        sat_pos:np.array,
                        sat_vel:np.array):

    # Rotate satellite positions in enu
    num_sats = np.ma.size(sat_pos,0)
    sat_pos_enu = []
    sat_vel_enu = []
    for count in range(num_sats):
        sat_pos_enu.append(conversions.ecef2enu(sat_pos[count,0],sat_pos[count,1],sat_pos[count,2],lla_0[0],lla_0[1],lla_0[2],deg=False))
        sat_vel_enu.append(pm.ecef2enuv(sat_vel[count,0],sat_vel[count,1],sat_vel[count,2],lla_0[0],lla_0[1],deg=False))

    sat_pos_enu = np.array(sat_pos_enu)
    sat_vel_enu = np.array(sat_vel_enu)
    
    # Turn EN into ENU
    current_pos = np.append(current_pos,0)
    current_vel = np.append(current_vel,0)

    # Unit Vectors and gemoetry, we'll see how the clock bias assumption goes
    u_vecs,_ = unit_vectors(current_pos,sat_pos_enu)

    del_y = []
    for count in range(num_sats):
        del_y.append(measured_doppler[count] - np.dot(u_vecs[count,:],sat_vel_enu[count,:]))

    # Estimate user velocities
    vel_est = np.linalg.pinv(-u_vecs[:,0:2])@del_y

    # Estimate heading
    heading = math.atan2(vel_est[0],vel_est[1])

    return heading



