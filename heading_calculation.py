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


def leo_pdr_heading_enu(lla_0:np.array,
                        measured_doppler:np.array,
                        current_pos:np.array,
                        current_vel:np.array,
                        sat_pos:np.array,
                        sat_vel:np.array):

    ellipsoid = pm.Ellipsoid(semimajor_axis=6378137.0,semiminor_axis=6356752.314245)

    # Rotate satellite positions in enu
    if sat_pos.ndim > 1:
        num_sats = np.ma.size(sat_pos,0)
    else:
        num_sats = 1
        sat_pos = sat_pos[...,np.newaxis].T
        sat_vel = sat_vel[...,np.newaxis].T

    sat_pos_enu = []
    sat_vel_enu = []
    for count in range(num_sats):
        sat_pos_enu.append(pm.ecef2enu(sat_pos[count,0],sat_pos[count,1],sat_pos[count,2],lla_0[0],lla_0[1],lla_0[2],ell=ellipsoid,deg=False))
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
    H = -u_vecs[:,0:2]
    vel_est = np.linalg.pinv(H)@del_y

    # Estimate heading
    heading = math.atan2(vel_est[0],vel_est[1])

    return heading



