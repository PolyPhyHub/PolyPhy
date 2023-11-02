from enum import IntEnum
import numpy as np
from numpy.random import default_rng
import taichi as ti

from utils.logger import Logger


class PPTypes:
    # TODO: impletement and test float 16 and 64
    # (int data don't occupy significant memory
    # so no need to optimize unless explicitly needed)
    FLOAT_CPU = np.float32
    FLOAT_GPU = ti.f32
    INT_CPU = np.int32
    INT_GPU = ti.i32

    VEC2i = ti.types.vector(2, INT_GPU)
    VEC3i = ti.types.vector(3, INT_GPU)
    VEC4i = ti.types.vector(4, INT_GPU)
    VEC2f = ti.types.vector(2, FLOAT_GPU)
    VEC3f = ti.types.vector(3, FLOAT_GPU)
    VEC4f = ti.types.vector(4, FLOAT_GPU)

    @staticmethod
    def set_precision(float_precision):
        if float_precision == "float64":
            PPTypes.FLOAT_CPU = np.float64
            PPTypes.FLOAT_GPU = ti.f64
        elif float_precision == "float32":
            PPTypes.FLOAT_CPU = np.float32
            PPTypes.FLOAT_GPU = ti.f32
        elif float_precision == "float16":
            PPTypes.FLOAT_CPU = np.float16
            PPTypes.FLOAT_GPU = ti.f16
        else:
            raise ValueError("Invalid float precision value. Supported values: \
                                float64, float32, float16")


class PPConfig:
    # TODO load trace resolution as argument
    # Distance sampling distribution for agents
    class EnumDistanceSamplingDistribution(IntEnum):
        CONSTANT = 0
        EXPONENTIAL = 1
        MAXWELL_BOLTZMANN = 2

    # Directional sampling distribution for agents
    class EnumDirectionalSamplingDistribution(IntEnum):
        DISCRETE = 0
        CONE = 1

    # Sampling strategy for directional agent mutation
    class EnumDirectionalMutationType(IntEnum):
        DETERMINISTIC = 0
        PROBABILISTIC = 1

    # Deposit fetching strategy
    class EnumDepositFetchingStrategy(IntEnum):
        NN = 0
        NN_PERTURBED = 1

    # Handling strategy for agents that leave domain boundary
    class EnumAgentBoundaryHandling(IntEnum):
        WRAP = 0
        REINIT_CENTER = 1
        REINIT_RANDOMLY = 2

    # State flags
    distance_sampling_distribution = EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
    directional_sampling_distribution = EnumDirectionalSamplingDistribution.CONE
    directional_mutation_type = EnumDirectionalMutationType.PROBABILISTIC
    deposit_fetching_strategy = EnumDepositFetchingStrategy.NN_PERTURBED
    agent_boundary_handling = EnumAgentBoundaryHandling.WRAP

    # Simulation-wide constants and defaults
    N_DATA_DEFAULT = 1000
    N_AGENTS_DEFAULT = 1000000
    DOMAIN_SIZE_DEFAULT = 100.0
    TRACE_RESOLUTION_MAX = 512
    DEPOSIT_DOWNSCALING_FACTOR = 1
    MAX_DEPOSIT = 10.0
    DOMAIN_MARGIN = 0.05
    RAY_EPSILON = 1.0e-3
    VIS_RESOLUTION = (1440, 900)

    # Input files
    input_file = ''

    # Simulation parameters
    sense_distance = 0.0
    sense_angle = 1.5
    steering_rate = 0.5
    step_size = 0.0
    sampling_exponent = 2.0
    deposit_attenuation = 0.9
    trace_attenuation = 0.96
    data_deposit = -1.0
    agent_deposit = -1.0
    deposit_vis = 0.1
    trace_vis = 1.0
    n_ray_steps = 50.0

    @staticmethod
    def set_value(constant_name, new_value):
        CONSTANT_VARS = ["N_DATA_DEFAULT", "N_AGENTS_DEFAULT", "DOMAIN_SIZE_DEFAULT"]
        if constant_name in CONSTANT_VARS:
            raise AssertionError("Changing const variables do not work!")
        if hasattr(PPConfig, constant_name):
            setattr(PPConfig, constant_name, new_value)
        else:
            raise AttributeError(f"'PPConfig' has no attribute '{constant_name}'")

    def setter(self, constant_name, new_value):
        if hasattr(self, constant_name):
            setattr(self, constant_name, new_value)
        else:
            raise AttributeError(f"'PPConfig' has no attribute '{constant_name}'")

    def register_data(self, ppData):
        pass

    def __init__(self):
        pass


class PPInputData:
    # TODO: determine ROOT automatically
    ROOT = '../../'
    input_file = ''

    def __load_from_file__(self):
        # Load from a file - parse file extension
        pass

    def __generate_test_data__(self, rng):
        # Load random data to test / simulation
        pass

    def __print_simulation_data_stats__(self):
        Logger.logToStdOut("info", 'Simulation domain min:', self.DOMAIN_MIN)
        Logger.logToStdOut("info", 'Simulation domain max:', self.DOMAIN_MAX)
        Logger.logToStdOut("info", 'Simulation domain size:', self.DOMAIN_SIZE)
        Logger.logToStdOut("info", 'Data sample:', self.data[0, :])
        Logger.logToStdOut("info", 'Number of agents:', self.N_AGENTS)
        Logger.logToStdOut("info", 'Number of data points:', self.N_DATA)

    def __init__(self, input_file, rng=default_rng()):
        # Initialize data and agents
        self.data = None
        self.DOMAIN_MIN = None
        self.DOMAIN_MAX = None
        self.DOMAIN_SIZE = None
        self.N_DATA = None
        self.N_AGENTS = None
        self.AVG_WEIGHT = None
        self.input_file = input_file
        if len(self.input_file) > 0:
            self.__load_from_file__()
        else:
            self.__generate_test_data__(rng)
        self.__print_simulation_data_stats__()


class PPInternalData:
    def __init_internal_data__(self, kernels):
        pass

    def edit_data(self, edit_index: PPTypes.INT_CPU,
                  window: ti.ui.Window) -> PPTypes.INT_CPU:
        pass

    def store_fit(self):
        pass

    def __init__(self, rng, kernels, ppConfig):
        pass


class PPSimulation:
    def __drawGUI__(self,
                    window,
                    ppConfig):
        pass

    def __init__(self, ppInternalData, ppConfig, batch_mode, num_iterations):
        pass


class PPPostSimulation:
    def __init__(self, ppInternalData):
        pass
