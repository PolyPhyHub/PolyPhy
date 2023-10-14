import os
import taichi as ti
from numpy.random import default_rng
from core.discrete2D import PPInputData_2DDiscrete, PPInternalData_2DDiscrete
from core.discrete2D import PPSimulation_2DDiscrete, PPPostSimulation_2DDiscrete
from kernel.discrete2D import PPKernels_2DDiscrete
from .common import PolyPhy


class PolyPhy_2DDiscrete(PolyPhy):
    def __init__(self, ppConfig):
        self.batch_mode = False
        self.num_iterations = -1
        self.ppConfig = ppConfig
        self.rng = default_rng()
        self.ppInputData = PPInputData_2DDiscrete(self.ppConfig.input_file, self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(arch=ti.cpu if os.path.exists("/tmp/flag") else ti.gpu)
        self.kernels = PPKernels_2DDiscrete()
        self.ppInternalData = PPInternalData_2DDiscrete(
            self.rng,
            self.kernels,
            self.ppConfig)

    def start_simulation(self):
        PPSimulation_2DDiscrete(
            self.ppInternalData,
            self.ppConfig,
            self.batch_mode,
            self.num_iterations)
        PPPostSimulation_2DDiscrete(self.ppInternalData)
