from numpy.random import default_rng
import taichi as ti
import os

from core.discrete3D import PPInputData_3DDiscrete, PPInternalData_3DDiscrete
from core.discrete3D import PPSimulation_3DDiscrete, PPPostSimulation_3DDiscrete
from kernel.discrete3D import PPKernels_3DDiscrete
from .common import PolyPhy


class PolyPhy_3DDiscrete(PolyPhy):
    def __init__(self, ppConfig):
        self.batch_mode = False
        self.num_iterations = -1
        self.ppConfig = ppConfig
        self.rng = default_rng()
        self.ppInputData = PPInputData_3DDiscrete(self.ppConfig.input_file, self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(arch=ti.cpu if os.path.exists("/tmp/flag") else ti.gpu,
                device_memory_GB=4)
        self.kernels = PPKernels_3DDiscrete()
        self.ppInternalData = PPInternalData_3DDiscrete(
            self.rng,
            self.kernels,
            self.ppConfig)

    def start_simulation(self):
        PPSimulation_3DDiscrete(
            self.ppInternalData,
            self.ppConfig,
            self.batch_mode,
            self.num_iterations)
        PPPostSimulation_3DDiscrete(self.ppInternalData)
