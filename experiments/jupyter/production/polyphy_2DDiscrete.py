from polyphy import *
import polyphy_core

class PolyPhy_2DDiscrete(PolyPhy):
    def __init__(self):
        self.parse_args()
        self.batch_mode = False
        self.num_iterations = -1
        self.input_file = ''
        self.ppConfig = PPConfig_2DDiscrete()
        self.parse_values()
        self.rng = default_rng()
        self.ppInputData = PPInputData_2DDiscrete(self.input_file,self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(arch=ti.gpu)
        self.kernels = Kernels()
        self.ppInternalData = PPInternalData(self.rng,self.kernels,self.ppConfig)

    def start_simulation(self):
        PPSimulation(self.ppInternalData,self.batch_mode,self.num_iterations)
        PPPostSimulation(self.ppInternalData)

PolyPhy_2DDiscrete().start_simulation()