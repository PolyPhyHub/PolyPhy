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
        self.ppInputData = PPInputData_2DDiscrete(self.input_file, self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(arch=ti.gpu)
        self.kernels = PPKernels()
        self.ppInternalData = PPInternalData_2DDiscrete(self.rng, self.kernels, self.ppConfig)

    def parse_args(self):
        super().parse_args()
        ## Place any additional arguments here, following the template in parent PolyPhy class
        self.args = self.parser.parse_args()

    def start_simulation(self):
        PPSimulation_2DDiscrete(self.ppInternalData, self.batch_mode, self.num_iterations)
        PPPostSimulation_2DDiscrete(self.ppInternalData)

PolyPhy_2DDiscrete().start_simulation()