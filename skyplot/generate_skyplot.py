# Code to generate the skyplot for the LEO PDR project
# Landon Boyd
# Tanner Koza
# 06/26/2024


import numpy as np
import navsim as ns
import matplotlib.pyplot as plt
import navtools as nt
import matplotlib.patheffects as pe
from collections.abc import Iterable
from collections import defaultdict

# Below are a bunkc of functions I stole from tannerkoza/eleventh_hour

def skyplot(
    az: np.ndarray, el: np.ndarray, name: str | list = None, deg: bool = True, **kwargs
):
    if isinstance(plt.gca(), plt.PolarAxes):
        ax = plt.gca()
    else:
        plt.close()
        fig = plt.gcf()
        ax = fig.add_subplot(projection="polar")

    if deg:
        az = np.radians(az)
    else:
        el = np.degrees(el)

    # format polar axes
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(91, 1)

    degree_sign = "\N{DEGREE SIGN}"
    r_labels = [
        "0" + degree_sign,
        "",
        "30" + degree_sign,
        "",
        "60" + degree_sign,
        "",
        "90" + degree_sign,
    ]
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=22.5)

    ax.set_axisbelow(True)

    # plot
    ax.scatter(az, el, **kwargs)

    # annotate object names
    if name is not None:
        if not isinstance(name, Iterable):
            name = (name,)

        for obj, n in enumerate(name):
            ax.annotate(
                n,
                (az[obj, 0], el[obj, 0]),
                fontsize="x-small",
                path_effects=[pe.withStroke(linewidth=3, foreground="w")],
            )

    ax.figure.canvas.draw()

    return ax

def create_skyplot(emitter_states: list):
    azimuth = defaultdict(lambda: defaultdict(lambda: []))
    elevation = defaultdict(lambda: defaultdict(lambda: []))

    for epoch in emitter_states:
        for emitter in epoch.values():
            azimuth[emitter.constellation][emitter.id].append(emitter.az)
            elevation[emitter.constellation][emitter.id].append(emitter.el)

    for constellation in azimuth.keys():
        az = np.array(nt.pad_list(list(azimuth[constellation].values())))
        el = np.array(nt.pad_list(list(elevation[constellation].values())))
        names = list(azimuth[constellation].keys())

        skyplot(
            az=az,
            el=el,
            label=constellation.lower(),
            name=names,
            deg=False,
            s=10,
        )



# Simulate constellation
rx_vel = np.array([0,0,0])
rx_pos = np.array([423756, -5361363, 3417705])

PROJECT_PATH = '/home/landon/Desktop/LEO_PDR/GPlus.yaml'
CONFIG_PATH ="."
DATA_PATH = PROJECT_PATH

configuration = ns.get_configuration(configuration_path=PROJECT_PATH)
sim = ns.get_signal_simulation(simulation_type="measurement", configuration=configuration)

sim.generate_truth(rx_pos=rx_pos.transpose(),rx_vel=rx_vel.transpose())
sim.simulate()
sat_states=sim.emitter_states


plt.figure()
ax = create_skyplot(sat_states.truth)
plt.show()




