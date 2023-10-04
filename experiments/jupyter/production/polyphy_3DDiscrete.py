from polyphy import *
import polyphy_core

class PolyPhy_3DDiscrete(PolyPhy):
    def __init__(self):
        self.parse_args()
        self.batch_mode = False
        self.num_iterations = -1
        self.input_file = ''
        self.ppConfig = PPConfig_3DDiscrete()
        self.parse_values()
        self.rng = default_rng()
        self.ppInputData = PPInputData_3DDiscrete(self.input_file, self.rng)
        self.ppConfig.register_data(self.ppInputData)
        ti.init(rch=ti.cpu if os.path.exists("/tmp/flag") else ti.gpu, device_memory_GB=4)
        self.kernels = PPKernels()
        self.ppInternalData = PPInternalData_3DDiscrete(self.rng, self.kernels, self.ppConfig)

    def parse_args(self):
        super().parse_args()
        ## Place any additional arguments here, following the template in parent PolyPhy class
        ## TODO add vis resolution as input argument
        self.args = self.parser.parse_args()

    def start_simulation(self):
        PPSimulation_3DDiscrete(self.ppInternalData, self.ppConfig, self.batch_mode, self.num_iterations)
        PPPostSimulation_3DDiscrete(self.ppInternalData)

PolyPhy_3DDiscrete().start_simulation()