import pandas as pd
import heading_calculation
import numpy as np
import pymap3d  as pm
import navsim as ns
from pathlib import Path
import matplotlib.pyplot as plt

# Waypoints for the user
time = np.array([1,2,3,4,5,6,7,8,9])
user_pos = np.array([[1,1,0],
                     [2,2,0],
                     [3,3,0],
                     [4,4,0],
                     [5,5,0],
                     [6,6,0],
                     [7,7,0],
                     [8,8,0],
                     [9,9,0]])
lla_0 = np.array([32,-85,200])

user_pos_ecef = np.array(pm.enu2ecef(user_pos[:,0],user_pos[:,1],user_pos[:,2],lla_0[0],lla_0[1],lla_0[2]))
user_vel_ecef = np.array(np.diff(user_pos_ecef,1,1)/np.diff(time))
user_vel = np.array(pm.ecef2enuv(user_vel_ecef[0,:],user_vel_ecef[1,:],user_vel_ecef[2,:],lla_0[0],lla_0[1])).T
lam = 299792458/1776e6

# Simulate with navsim
PROJECT_PATH = Path(__file__).parent
CONFIG_PATH ="."
DATA_PATH = PROJECT_PATH
configuration = ns.get_configuration(configuration_path=PROJECT_PATH)
sim = ns.get_signal_simulation(simulation_type="measurement", configuration=configuration)
sim.generate_truth(rx_pos=user_pos_ecef.T,rx_vel=user_vel_ecef.T)
sim.simulate()
observables = sim.observables
sat_states = sim.emitter_states
ll = np.radians(lla_0[0:2])
lla_0 = np.append(ll,lla_0[2])

# Test the function
count = 0
leo_heading = []

for sat_state,observable in zip(sat_states.truth,observables):
    all_pos = np.squeeze(np.array([emitter_state.pos for emitter_state in sat_state.values()]))
    all_vel = np.squeeze(np.array([emitter_state.vel for emitter_state in sat_state.values()]))

    all_dopp = np.array([observable[sat].pseudorange_rate for sat in observable.keys()])

    user_pos_en = user_pos[count,0:2]
    user_vel_en = user_vel[count,0:2]
    leo_heading.append(heading_calculation.leo_pdr_heading_enu(lla_0,all_dopp,user_pos_en,user_vel_en,all_pos,all_vel))

    count = count + 1


leo_heading = np.array(leo_heading)

plt.figure()
plt.plot(leo_heading)
plt.show()











