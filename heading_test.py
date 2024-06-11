import pandas as pd
import heading_calculation
import numpy as np


sat_pos = np.array([[1000,1000,1000],[2000,2000,2000]])
sat_vel = np.array([[100,0,0],[0,100,0]])
user_pos = np.array([0,0,0])
user_vel = np.array([10,0,0])
user_lla = np.array([32,-85,200])
doppler = np.array([10,10])

# Test the function
heading_calculation.leo_pdr_heading(user_lla,doppler,user_pos,user_vel,sat_pos,sat_vel)






