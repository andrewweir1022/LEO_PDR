# Function to take in Iridium STL observables and generate a heading in the NED frame
# Landon Boyd
# 06/10/2024

import numpy as np
import navtools.conversions as conversions

def leo_pdr_heading(lla_0:np.array,
                    measured_doppler:np.array,
                    current_pos:np.array,
                    sat_pos:np.array):
    
    # Assumption that user is not flying
    current_ecef = conversions.enu2ecef(current_pos[0],current_pos[1],0,lla_0[0],lla_0[1],lla_0[2])


    pass