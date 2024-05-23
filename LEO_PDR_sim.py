import yaml
from pathlib import Path
import numpy as np
import pymap3d as pm
from scipy.io import loadmat
from scipy.signal import sosfiltfilt, butter
from scipy import interpolate

#Define Monte Carlo Configuration
scriptPath = Path(__file__).parent.resolve()
configPath = Path(scriptPath, "sim_config.yml")

with open(str(configPath), "r") as configFile:
    try:
        # parse config file
        config = yaml.safe_load(configFile)
    except yaml.YAMLError as exc:
        print(exc)

data_path: str =config["PathToInput"]

input_data=loadmat(data_path,struct_as_record=False)

lat=np.deg2rad(input_data["Lat"]).transpose()
lon=np.deg2rad(input_data["Lon"]).transpose()
truth_time=input_data["time"].squeeze()

#smooth data to create a truth path

sos = butter(4, 0.009, output="sos")
truth_lat = sosfiltfilt(sos, lat).transpose()
truth_lon = sosfiltfilt(sos, lon).transpose()

truth_enu = np.asarray(pm.geodetic2enu(truth_lat,truth_lon,np.zeros_like(truth_lat),truth_lat[0],truth_lon[0],0,ell=pm.Ellipsoid.from_name('wgs84'),deg=False)).squeeze()
truth_east=truth_enu[0,:]
truth_north=truth_enu[0,:]

InterpolatorE = interpolate.interp1d(truth_time, truth_east, fill_value="extrapolate")
InterpolatorN = interpolate.interp1d(truth_time, truth_north, fill_value="extrapolate")

time_upsampled=np.linspace(truth_time[0],truth_time[-1],len(truth_time)*100)
east_upsampled=InterpolatorE(time_upsampled)
north_upsampled=InterpolatorN(time_upsampled)

east_step=np.empty([1,len(east_upsampled)])
north_step=np.empty([1,len(north_upsampled)])

delta_east=np.diff(east_upsampled)
delta_north=np.diff(north_upsampled)

total_dist=0
for ii in range(len(time_upsampled)-1):
    delta_dist=np.sqrt(delta_east[ii]**2+delta_north[ii]**2)
    total_dist=total_dist+delta_dist
    if total_dist>=0.7:
        east_step[0,ii]=east_upsampled[ii]
        north_step[0,ii]=north_upsampled[ii]
        total_dist=0

p=1