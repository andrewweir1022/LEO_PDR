# Function to take in Iridium STL observables and generate a heading in the NED frame
# Landon Boyd
# 06/10/2024

import numpy as np
import navtools.conversions as conversions

def unit_vectors(rx_pos:np.ndarray,sat_pos:np.ndarray):
    # Function to extract the unit vectors and geometric range estimates for GPS positioning
    # Landon Boyd
    
    sat_rel_pos = sat_pos - rx_pos
    ranges = np.linalg.norm(sat_rel_pos)
    unit_vectors = sat_rel_pos / ranges
    return unit_vectors,ranges

def leo_pdr_heading(lla_0:np.array,
                    measured_doppler:np.array,
                    current_pos:np.array,
                    current_vel:np.array,
                    sat_pos:np.array,
                    sat_vel:np.array):
    
    # 0 vertical pos and vel
    current_ecef = conversions.enu2ecef(current_pos[0],current_pos[1],0,lla_0[0],lla_0[1],lla_0[2])
    current_ecef_vel = conversions.enu2uvw(current_vel[0],current_vel[1],0,lla_0[0],lla_0[1])
    
    # Unit Vectors
    u_vecs,ranges = unit_vectors(current_ecef,sat_pos)

    pass