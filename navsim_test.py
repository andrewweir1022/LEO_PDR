# %%
import numpy as np
import navsim as ns

from pathlib import Path

PROJECT_PATH = Path(__file__).parent
CONFIG_PATH ="."
DATA_PATH = PROJECT_PATH / "data"

rx_pos = np.array([423756, -5361363, 3417705])

configuration = ns.get_configuration(configuration_path=PROJECT_PATH)
sim = ns.get_signal_simulation(
    simulation_type="measurement", configuration=configuration
)

sim.generate_truth(rx_pos=rx_pos)
sim.simulate()
observables = sim.observables
#sim.to_hdf(output_dir_path=DATA_PATH)
