# %%
import yaml
from pathlib import Path
import numpy as np
import pymap3d as pm
from scipy.io import loadmat
from scipy.signal import sosfiltfilt, butter
from scipy import interpolate
import matplotlib.pyplot as plt

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
truth_time=input_data["time"].squeeze()

#smooth LLA data to create a truth path
sos = butter(4, 0.05, output="sos")
truth_lat = sosfiltfilt(sos, lat).transpose()
truth_lon = sosfiltfilt(sos, lon).transpose()

#convert truth LLA to truth ENU
truth_enu = np.asarray(pm.geodetic2enu(truth_lat,truth_lon,np.zeros_like(truth_lat),truth_lat[0],truth_lon[0],0,ell=pm.Ellipsoid.from_name('wgs84'),deg=False)).squeeze()
truth_east=truth_enu[0,:]
truth_north=truth_enu[1,:]

#upsample position values to decrease size of delta positions
InterpolatorE = interpolate.interp1d(truth_time, truth_east, fill_value="extrapolate")
InterpolatorN = interpolate.interp1d(truth_time, truth_north, fill_value="extrapolate")

time_upsampled=np.linspace(truth_time[0],truth_time[-1],len(truth_time)*100)
east_upsampled=InterpolatorE(time_upsampled)
north_upsampled=InterpolatorN(time_upsampled)

delta_east=np.diff(east_upsampled)
delta_north=np.diff(north_upsampled)

#define variables at step instances
east_step=np.zeros([1,len(east_upsampled)])
north_step=np.zeros([1,len(north_upsampled)])
time_step=np.zeros([1,len(north_upsampled)])

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
        total_dist=0

#reduce larger matricies to onlt step instance measurments
east_step[0,0]=1
north_step[0,0]=1
time_step[0,0]=1
east_step = east_step[np.not_equal(east_step,0)]
north_step = north_step[np.not_equal(north_step,0)]
time_step = time_step[np.not_equal(time_step,0)]
east_step[0]=0
north_step[0]=0
time_step[0]=0

#initialize variables for truth measurment creation
del_east=np.diff(east_step,prepend=0)
del_north=np.diff(north_step,prepend=0)
step_length_truth=np.zeros([1,len(east_step)])
heading_truth=np.zeros([1,len(east_step)])
dt=np.diff(time_step,prepend=0)
vel_e=np.zeros([1,len(east_step)])
vel_n=np.zeros([1,len(east_step)])

#loop through each position at step instance
for ii in range(len(east_step)):

    #create step length truth
    step_length_truth[0,ii]=np.sqrt(del_east[ii]**2+del_north[ii]**2)
    #create heading truth (rad)
    heading_truth[0,ii]=np.arctan2(del_east[ii], del_north[ii])

    #create delta positions
    delta_e = step_length_truth[0,ii] * np.sin(heading_truth[0,ii])
    delta_n = step_length_truth[0,ii] * np.cos(heading_truth[0,ii])
    
    #convert delta positions into velocities
    if ii>0:
        vel_e[0,ii]=delta_e/dt[ii]
        vel_n[0,ii]=delta_n/dt[ii]

#put noise on step length values to create measurments
SL_sigma: str =config["StepLengthNoise"]
SL_noise = np.random.normal(0, SL_sigma, step_length_truth.size)
SL_measurment=step_length_truth+SL_noise

#put noise and bias on heading values to create measurements
heading_sigma: str =config["HeadingNoise"]
heading_noise = np.random.normal(0, np.deg2rad(heading_sigma), heading_truth.size)

heading_tau: str =config["HeadingBiasTau"]
heading_bias_sigma: str =config["HeadingBiasNoise"]
heading_bias=np.zeros([1,np.size(heading_truth)])

#bias is estimated as FOGM
for ii in range(np.size(heading_truth)-1):
    heading_bias[0,ii+1]=np.exp(-(1/heading_tau)*dt[ii])*heading_bias[0,ii]+np.random.normal(0, np.deg2rad(heading_bias_sigma), 1)

heading_measurment=heading_truth+heading_noise+heading_bias

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
