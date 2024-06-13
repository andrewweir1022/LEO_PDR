# %%
import yaml
from pathlib import Path
import numpy as np
import pymap3d as pm
from scipy.io import loadmat
from scipy.signal import sosfiltfilt, butter
from scipy import interpolate
import matplotlib.pyplot as plt
import navsim as ns
# import DEM_call as DM
# import pyhigh as ph
# import pvlib.location as pv
# import Baro_UKF as baro
import pandas as pd
from heading_calculation import leo_pdr_heading_enu, leo_pdr_heading
import navtools.conversions as conversions

#Open Config file
scriptPath = Path(__file__).parent.resolve()
configPath = Path(scriptPath, "sim_config.yml")

with open(str(configPath), "r") as configFile:
    try:
        # parse config file
        config = yaml.safe_load(configFile)
    except yaml.YAMLError as exc:
        print(exc)

#define path to input .mat file
data_path: str =config["PathToInput"]

#open .mat input file and pull values
input_data=loadmat(data_path,struct_as_record=False)

lat=np.deg2rad(input_data["Lat"]).transpose()
lon=np.deg2rad(input_data["Lon"]).transpose()
alt=input_data["Alt"]
truth_time=input_data["time"].squeeze()

#smooth LLA data to create a truth path
sos = butter(4, 0.05, output="sos")
truth_lat = sosfiltfilt(sos, lat).transpose()
truth_lon = sosfiltfilt(sos, lon).transpose()

#convert truth LLA to truth ENU
truth_enu = np.asarray(pm.geodetic2enu(truth_lat,truth_lon,alt,truth_lat[0],truth_lon[0],alt[0],ell=pm.Ellipsoid.from_name('wgs84'),deg=False)).squeeze()
truth_east=truth_enu[0,:]
truth_north=truth_enu[1,:]
truth_up=truth_enu[2,:]

truth_ecef=np.asarray(pm.geodetic2ecef(truth_lat,truth_lon,alt,ell=pm.Ellipsoid.from_name('wgs84'),deg=False)).squeeze()
truth_x=truth_ecef[0,:]
truth_y=truth_ecef[1,:]
truth_z=truth_ecef[2,:]

#upsample position values to decrease size of delta positions
InterpolatorE = interpolate.interp1d(truth_time, truth_east, fill_value="extrapolate")
InterpolatorN = interpolate.interp1d(truth_time, truth_north, fill_value="extrapolate")

time_upsampled=np.linspace(truth_time[0],truth_time[-1],len(truth_time)*100)
east_upsampled=InterpolatorE(time_upsampled)
north_upsampled=InterpolatorN(time_upsampled)

delta_east=np.diff(east_upsampled)
delta_north=np.diff(north_upsampled)

sos2 = butter(4, 0.001, output="sos")
delta_east = sosfiltfilt(sos2, delta_east)
delta_north = sosfiltfilt(sos2, delta_north)
# %%

#define variables at step instances
east_step=np.empty([1,len(east_upsampled)])
east_step[:]=np.nan
north_step=np.empty([1,len(north_upsampled)])
north_step[:]=np.nan
time_step=np.empty([1,len(time_upsampled)])
time_step[:]=np.nan
total_dist_stg=np.empty([1,len(north_upsampled)])
total_dist_stg[:]=np.nan

#loop through upsampled positions to create step length chuncks
total_dist=0
for ii in range(len(time_upsampled)-1):

    #determine total horizontal distance from positions
    delta_dist=np.sqrt(delta_east[ii]**2+delta_north[ii]**2)
    total_dist=total_dist+delta_dist

    #when total distance eclipses step length chunck, define values
    if total_dist>=0.7:
        east_step[0,ii]=east_upsampled[ii]
        north_step[0,ii]=north_upsampled[ii]
        time_step[0,ii]=time_upsampled[ii]
        total_dist_stg[0,ii]=total_dist
        total_dist=0

#reduce larger matricies to only step instance measurments
east_step = east_step[~np.isnan(east_step)]
north_step = north_step[~np.isnan(north_step)]
time_step = time_step[~np.isnan(time_step)]
total_dist_step=total_dist_stg[~np.isnan(total_dist_stg)]
elevation_step=np.zeros([1,len(east_step)])

# for ii in range(len(east_step)):
#     LLA_step=np.asarray(pm.enu2geodetic(east_step[ii],north_step[ii],0,truth_lat[0][0],truth_lon[0][0],alt[0][0],ell=pm.Ellipsoid.from_name('wgs84'),deg=False)).squeeze()
#     #elevation_step[0][ii]=DM.elevation_google(np.rad2deg(LLA_step[0]),np.rad2deg(LLA_step[1]))
#     elevation_step[0][ii]=pv.lookup_altitude(np.rad2deg(LLA_step[0]),np.rad2deg(LLA_step[1]))

#initialize variables for truth measurment creation
del_east_step=np.diff(east_step,prepend=0)
del_north_step=np.diff(north_step,prepend=0)
del_east_step = sosfiltfilt(sos, del_east_step)
del_north_step = sosfiltfilt(sos, del_north_step)
step_length_truth=np.zeros([1,len(east_step)])
heading_truth=np.zeros([1,len(east_step)])
dt=np.diff(time_step,prepend=0)
vel_e_truth=np.zeros([1,len(east_step)])
vel_n_truth=np.zeros([1,len(east_step)])

# %%
#loop through each position at step instance
for ii in range(len(east_step)):

    #create step length truth
    step_length_truth[0,ii]=np.sqrt(del_east_step[ii]**2+del_north_step[ii]**2)
    #create heading truth (rad)
    heading_truth[0,ii]=np.arctan2(del_east_step[ii], del_north_step[ii])

    #create delta positions
    delta_e = step_length_truth[0,ii] * np.sin(heading_truth[0,ii])
    delta_n = step_length_truth[0,ii] * np.cos(heading_truth[0,ii])
    
    #convert delta positions into velocities
    vel_e_truth[0,ii]=delta_e/dt[ii]
    vel_n_truth[0,ii]=delta_n/dt[ii]

# %%
#put noise on step length values to create measurments
SL_sigma: str =config["StepLengthNoise"]
SL_noise = np.random.normal(0, SL_sigma, step_length_truth.size)
SL_measurment=step_length_truth+SL_noise

#put noise and bias on heading values to create measurements
heading_sigma: str =config["HeadingNoise"]
heading_noise = np.random.normal(0, np.deg2rad(heading_sigma), heading_truth.size)

heading_tau: str =config["HeadingBiasTau"]
heading_bias_sigma: str =config["HeadingBiasNoise"]
heading_init: str =config["HeadingInit"]
heading_bias=np.zeros([1,np.size(heading_truth)])

#bias is estimated as FOGM
for ii in range(np.size(heading_truth)-1):
    heading_bias[0,ii+1]=np.exp(-(1/heading_tau)*dt[ii])*heading_bias[0,ii]+np.random.normal(0, np.deg2rad(heading_bias_sigma), size=None)

heading_measurment=heading_truth+heading_noise+heading_bias+np.deg2rad(heading_init)

#put noise and bias on barometer values to create measurements
barometer_sigma: str =config["BarometerNoise"]
barometer_noise = np.random.normal(0, barometer_sigma, elevation_step.size)

barometer_tau: str =config["BarometerBiasTau"]
barometer_bias_sigma: str =config["BarometerBiasNoise"]
barometer_bias=np.zeros([1,np.size(elevation_step)])

#bias is estimated as FOGM
for ii in range(np.size(elevation_step)-1):
    barometer_bias[0,ii+1]=np.exp(-(1/barometer_tau)*dt[ii])*barometer_bias[0,ii]+np.random.normal(0, barometer_bias_sigma, size=None)

barometer_measurment=elevation_step+barometer_noise+barometer_bias

df_time=pd.DataFrame(time_step.transpose(),columns=['time'])
df_heading=pd.DataFrame(heading_measurment.transpose(),columns=['heading'])
df_SL=pd.DataFrame(SL_measurment.transpose(),columns=['step_length'])
df_dt=pd.DataFrame(dt.transpose(),columns=['PDR_dt'])
df_PDR=pd.concat([df_time,df_heading,df_SL,df_dt],axis=1)

init_LLA=[truth_lat[0][0],truth_lon[0][0],alt[0][0]]
# states_UKF=baro.UKF_Run(SL_measurment,heading_measurment,barometer_measurment,dt,init_LLA)

#initialize raw position measurments
raw_east=np.zeros([1,np.size(step_length_truth)+1])
raw_north=np.zeros([1,np.size(step_length_truth)+1])

#create raw position measurments
for ii in range(np.size(step_length_truth)):
    raw_east[0,ii+1]=raw_east[0,ii]+SL_measurment[0,ii]*np.sin(heading_measurment[0,ii])
    raw_north[0,ii+1]=raw_north[0,ii]+SL_measurment[0,ii]*np.cos(heading_measurment[0,ii])

plt.figure()
plt.plot(raw_east[0,:],raw_north[0,:])
plt.plot(truth_east,truth_north)
# plt.plot(states_UKF[1,:],states_UKF[0,:])
plt.legend(['raw','truth','UKF'])
# %%

ecef_step=np.zeros((3,len(east_step)))
for ii in range((len(east_step))):
    ecef_step[:,ii]=conversions.enu2ecef(east_step[ii],north_step[ii],0,init_LLA[0],init_LLA[1],init_LLA[2],deg=False)

init_ecef=conversions.enu2ecef(0,0,0,init_LLA[0],init_LLA[1],init_LLA[2],deg=False)
#difference position measurments ECEF
delta_x=np.diff(ecef_step[0,:],prepend=init_ecef[0])
delta_y=np.diff(ecef_step[1,:],prepend=init_ecef[1])
delta_z=np.diff(ecef_step[2,:],prepend=init_ecef[2])

#calculate velocities
vel_x=np.divide(delta_x,dt)
vel_y=np.divide(delta_y,dt)
vel_z=np.divide(delta_z,dt)

vel_x = sosfiltfilt(sos, vel_x)
vel_y = sosfiltfilt(sos, vel_y)
vel_z = sosfiltfilt(sos, vel_z)
# %%
#use navsim to generate LEO observables
rx_pos=ecef_step
rx_vel=np.asarray([vel_x,vel_y,vel_z])

#setting up navsim
PROJECT_PATH = Path(__file__).parent
CONFIG_PATH ="."
DATA_PATH = PROJECT_PATH

configuration = ns.get_configuration(configuration_path=PROJECT_PATH)
sim = ns.get_signal_simulation(simulation_type="measurement", configuration=configuration)

#pulling observables
sim.generate_truth(rx_pos=rx_pos.transpose(),rx_vel=rx_vel.transpose())
sim.simulate()
observables = sim.observables
sat_states=sim.emitter_states

sats_in_view=[]
#detrmining maximum number of LEO observables in run
for ii in range((len(observables))):
    ind_observables=observables[ii]
    ind_keys=list(ind_observables)
    for kk in range((len(ind_observables))):
        sats_in_view.append(ind_keys[kk])

sats_in_view = list(dict.fromkeys(sats_in_view))
#initializing LEO variables
num_sats_max=len(sats_in_view)
doppler=np.empty((num_sats_max,len(observables)))
doppler[:]=np.nan
sat_pos=np.empty((num_sats_max,len(observables),3))
sat_pos[:]=np.nan
sat_vel=np.empty((num_sats_max,len(observables),3))
sat_vel[:]=np.nan
Hz=1
Dt=1/Hz
LEO_time=np.ones((1,len(observables)+1))
LEO_time[0,0]=time_step[0]
ephemeris=sat_states.ephemeris

#pulling doppler, sat positions, sat velocities
for ii in range(len(observables)):
    ind_observables=observables[ii]
    keys=list(ind_observables)
    ephem_obs=ephemeris[ii]
    for kk in range((len(keys))):
        sat=ind_observables[keys[kk]]
        lam=(299792458)/(1776e6)
        idx=sats_in_view.index(keys[kk])
        doppler[idx,ii]=(sat.carrier_doppler*-lam)
        Sat_state=ephem_obs[keys[kk]]
        sat_pos[idx,ii,:]=Sat_state.pos
        sat_vel[idx,ii,:]=Sat_state.vel
    LEO_time[0,ii+1]=LEO_time[0,ii]+Dt
LEO_time=np.delete(LEO_time,-1)

#creating LEO DF
df_time_LEO=pd.DataFrame(LEO_time.transpose(),columns=['time'])
df_doppler=pd.DataFrame()
for ii in range((num_sats_max)):
    doppler_idx=pd.DataFrame(doppler[ii,:].transpose(),columns=[sats_in_view[ii]])
    df_doppler=pd.concat([df_doppler,doppler_idx],axis=1)
df_LEO=pd.concat([df_time_LEO,df_doppler],axis=1)

#ordering all input measurments
input_data=pd.merge_ordered(df_PDR,df_LEO, on="time")
input_data=input_data.to_numpy()

#initializing storage variables
heading_residual=heading_measurment[0,0]-heading_truth[0,0]
n=1
i=0
sat_index=np.linspace(0,(num_sats_max)-1,(num_sats_max))
East=0
North=0
East_stg=np.zeros((1,len(time_step)+1))
North_stg=np.zeros((1,len(time_step)+1))
leo_heading_stg=np.zeros((1,len(LEO_time)))

#looping through input data
for ii in range((len(input_data))):
    #detected PDR measurement
    if ~np.isnan(input_data[ii,1]):
        #delta positions
        delta_E=input_data[ii,2] * np.sin(input_data[ii,1]-heading_residual)
        delta_N=input_data[ii,2] * np.cos(input_data[ii,1]-heading_residual)

        #velcoity
        Vel_E=delta_E/input_data[ii,3]
        Vel_N=delta_N/input_data[ii,3]
        if np.isnan(Vel_E):
            Vel_E=0
            Vel_N=0

        #positions
        East=East+delta_E
        North=North+delta_N
        East_stg[0,i+1]=East
        North_stg[0,i+1]=North
        i=i+1

        #last heading value
        heading_stg=input_data[ii,1]

    #detected LEO measurment
    if ~np.isnan(input_data[ii,4]) and ii>2:

        #pull all dopplers in view
        dopps=input_data[ii,4:]
        sat_idx=sat_index[~np.isnan(dopps)]
        dopps=np.delete(dopps,np.isnan(dopps))

        #pull corresponding sat positions and velocities
        dopp_pos=np.zeros((len(sat_idx),3))
        dopp_vel=np.zeros((len(sat_idx),3))
        for kk in range((len(sat_idx))):
            dopp_pos[kk,:]=sat_pos[int(sat_idx[kk]),n,:]
            dopp_vel[kk,:]=sat_vel[int(sat_idx[kk]),n,:]

        #pull user positions and velocities
        user_pos=np.asarray([East,North])
        user_vel=np.asarray([Vel_E,Vel_N])
        if n==25:
            o=1
        leo_heading=leo_pdr_heading_enu(init_LLA,dopps,user_pos,user_vel,dopp_pos,dopp_vel)
        leo_heading_stg[0,n]=leo_heading
        # heading_residual=heading_stg-leo_heading
        n=n+1

leo_heading_stg[0,0]=heading_truth[0,0]
plt.plot(East_stg[0,:],North_stg[0,:])

plt.figure()
plt.plot(LEO_time,np.rad2deg(leo_heading_stg[0,:]))
plt.plot(time_step,np.rad2deg(heading_truth[0,:]))
plt.plot(time_step,np.rad2deg(heading_measurment[0,:]))

# %%
