# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import os
import taichi as ti
from numpy.random import default_rng
from core.continuous2D import PPInputData_2DContinuous, PPInternalData_2DContinuous
from core.continuous2D import PPSimulation_2DContinuous, PPPostSimulation_2DContinuous
from kernel.continuous2D import PPKernels_2DContinuous
from .common import PolyPhy


class PolyPhy_2DContinuous(PolyPhy):
    def __init__(self, ppConfig, batch_mode, num_iterations):
        self.batch_mode = batch_mode
        self.num_iterations = num_iterations
        self.ppConfig = ppConfig
        self.rng = default_rng()
        self.ppInputData = PPInputData_2DContinuous(self.ppConfig.input_file, self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(arch=ti.cpu)
        self.kernels = PPKernels_2DContinuous()
        self.ppInternalData = PPInternalData_2DContinuous(
            self.rng,
            self.kernels,
            self.ppConfig)

    def start_simulation(self):
        PPSimulation_2DContinuous(
            self.ppInternalData,
            self.ppConfig,
            self.batch_mode,
            self.num_iterations)
        PPPostSimulation_2DContinuous(self.ppInternalData)
