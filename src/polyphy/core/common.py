# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import os
from enum import IntEnum
import numpy as np
from numpy.random import default_rng
import taichi as ti

from utils.logger import Logger


class PPTypes:
    # TODO: impletement and test float 16 and 64
    # (int data don't occupy significant memory
    # so no need to optimize unless explicitly needed)
    """
    PPTypes: Defines various data types used in PolyPhy library.

    Attributes:
        - FLOAT_CPU: Numpy float type for CPU.
        - FLOAT_GPU: Taichi float type for GPU.
        - INT_CPU: Numpy integer type for CPU.
        - INT_GPU: Taichi integer type for GPU.
        - VEC2i: Taichi vector type with 2 integers.
        - VEC3i: Taichi vector type with 3 integers.
        - VEC4i: Taichi vector type with 4 integers.
        - VEC2f: Taichi vector type with 2 floats.
        - VEC3f: Taichi vector type with 3 floats.
        - VEC4f: Taichi vector type with 4 floats.

    Methods:
        - set_precision(float_precision): Set the float precision for the library.

    """
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
        """
        Set the float precision for the library.

        Args:
            - float_precision (str): The desired float precision. Supported values: "float64", "float32", "float16".

        Raises:
            - ValueError: If an invalid float precision value is provided.

        """
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
            raise ValueError("Invalid float precision value. Supported values: float64, float32, float16")


class PPConfig:
    """
    PPConfig: Configuration Class for PolyPhy Library.

    Nested Classes:
        - EnumDistanceSamplingDistribution: Enumerated values for distance sampling distributions.
        - EnumDirectionalSamplingDistribution: Enumerated values for directional sampling distributions.
        - EnumDirectionalMutationType: Enumerated values for directional mutation types.
        - EnumDepositFetchingStrategy: Enumerated values for deposit fetching strategies.
        - EnumAgentBoundaryHandling: Enumerated values for agent boundary handling strategies.

    """
    # TODO load trace resolution as argument
    # Distance sampling distribution for agents
    class EnumDistanceSamplingDistribution(IntEnum):
        """
        EnumDistanceSamplingDistribution: Enumerated values for distance sampling distributions.

        Values:
            - CONSTANT: Constant distance sampling distribution.
            - EXPONENTIAL: Exponential distance sampling distribution.
            - MAXWELL_BOLTZMANN: Maxwell-Boltzmann distance sampling distribution.

        """
        CONSTANT = 0
        EXPONENTIAL = 1
        MAXWELL_BOLTZMANN = 2

    # Directional sampling distribution for agents
    class EnumDirectionalSamplingDistribution(IntEnum):
        """
        EnumDirectionalSamplingDistribution: Enumerated values for directional sampling distributions.

        Values:
            - DISCRETE: Discrete directional sampling distribution.
            - CONE: Cone directional sampling distribution.

        """
        DISCRETE = 0
        CONE = 1

    # Sampling strategy for directional agent mutation
    class EnumDirectionalMutationType(IntEnum):
        """
        EnumDirectionalMutationType: Enumerated values for directional mutation types.

        Values:
            - DETERMINISTIC: Deterministic directional mutation type.
            - PROBABILISTIC: Probabilistic directional mutation type.

        """
        DETERMINISTIC = 0
        PROBABILISTIC = 1

    # Deposit fetching strategy
    class EnumDepositFetchingStrategy(IntEnum):
        """
        EnumDepositFetchingStrategy: Enumerated values for deposit fetching strategies.

        Values:
            - NN: Nearest neighbor deposit fetching strategy.
            - NN_PERTURBED: Perturbed nearest neighbor deposit fetching strategy.

        """
        NN = 0
        NN_PERTURBED = 1

    # Handling strategy for agents that leave domain boundary
    class EnumAgentBoundaryHandling(IntEnum):
        """
        EnumAgentBoundaryHandling: Enumerated values for agent boundary handling strategies.

        Values:
            - WRAP: Wrap-around boundary handling strategy.
            - REINIT_CENTER: Reinitialize at the center of the domain boundary handling strategy.
            - REINIT_RANDOMLY: Reinitialize randomly within the domain boundary handling strategy.

        """
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
    VIS_RESOLUTION = (1280, 720)

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
        """
        Set the value of a constant variable in the class.

        Args:
            - constant_name (str): The name of the constant variable.
            - new_value: The new value to set for the constant variable.

        Raises:
            - AssertionError: If attempting to change predefined constants.
            - AttributeError: If the specified constant does not exist in the class.

        """
        CONSTANT_VARS = ["N_DATA_DEFAULT", "N_AGENTS_DEFAULT", "DOMAIN_SIZE_DEFAULT"]
        if constant_name in CONSTANT_VARS:
            raise AssertionError("Changing const variables do not work!")
        if hasattr(PPConfig, constant_name):
            setattr(PPConfig, constant_name, new_value)
        else:
            raise AttributeError(f"'PPConfig' has no attribute '{constant_name}'")

    def setter(self, constant_name, new_value):
        """
        Instance method to set the value of a constant variable.

        Args:
            - constant_name (str): The name of the constant variable.
            - new_value: The new value to set for the constant variable.

        Raises:
            - AttributeError: If the specified constant does not exist in the instance.

        """
        if hasattr(self, constant_name):
            setattr(self, constant_name, new_value)
        else:
            raise AttributeError(f"'PPConfig' has no attribute '{constant_name}'")

    def register_data(self, ppData):
        """
        Placeholder method for registering data (not implemented).

        Args:
            - ppData: Placeholder argument for data registration.

        """
        pass

    def __init__(self):
        """
        Initializes an instance of the PPConfig class.

        """
        pass


class PPInputData:
    """
    PPInputData: Input Data Class for PolyPhy Library.

    Methods:
        - __load_from_file__(): Load data from a file and parse the file extension (not implemented).
        - __generate_test_data__(rng): Load random data for testing or simulation.
        - __print_simulation_data_stats__(): Print statistics of the simulation data.
        - __init__(input_file, rng=default_rng()): Initialize an instance of PPInputData with input data from a file or
          generated test data.

    Attributes:
        - ROOT: Root directory path for file operations.
        - input_file: Path to the input file.

    """
    # TODO: determine ROOT automatically
    ROOT = '../../'
    input_file = ''

    def __load_from_file__(self):
        """
        Load data from a file and parse the file extension (not implemented).

        """
        # Load from a file - parse file extension
        pass

    def __generate_test_data__(self, rng):
        """
        Load random data for testing or simulation.

        Args:
            - rng: Random number generator for data generation.

        """
        # Load random data to test / simulation
        pass

    def __print_simulation_data_stats__(self):
        """
        Print statistics of the simulation data.

        """
        Logger.logToStdOut("info", 'Simulation domain min:', self.DOMAIN_MIN)
        Logger.logToStdOut("info", 'Simulation domain max:', self.DOMAIN_MAX)
        Logger.logToStdOut("info", 'Simulation domain size:', self.DOMAIN_SIZE)
        # Logger.logToStdOut("info", 'Data sample:', self.data[0, :])
        Logger.logToStdOut("info", 'Number of agents:', self.N_AGENTS)
        Logger.logToStdOut("info", 'Number of data points:', self.N_DATA)

    def __init__(self, input_file, rng=default_rng()):
        """
        Initialize an instance of PPInputData with input data from a file or generated test data.

        Args:
            - input_file: Path to the input file.
            - rng: Random number generator for data generation.

        """
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
    """
    PPInternalData: Internal Data Class for PolyPhy Library.

    Methods:
        - __init_internal_data__(kernels): Initialize internal data with specified kernels.
        - edit_data(edit_index, window): Edit internal data based on the edit index and GUI window.
        - store_fit(): Store fit solutions in the 'data/fits/' directory.
        - __init__(rng, kernels, ppConfig): Initialize an instance of PPInternalData with specified random number
          generator, kernels, and configuration.

    Attributes:
        - (Attributes specific to the internal data)

    """
    def __init_internal_data__(self, kernels):
        """
        Initialize internal data with specified kernels.

        Args:
            - kernels: Kernels for internal data initialization.

        """
        pass

    def edit_data(self, edit_index: PPTypes.INT_CPU,
                  window: ti.ui.Window) -> PPTypes.INT_CPU:
        """
        Edit internal data based on the edit index and GUI window.

        Args:
            - edit_index (PPTypes.INT_CPU): Index for editing data.
            - window (ti.ui.Window): GUI window for editing.

        Returns:
            - PPTypes.INT_CPU: Updated index after editing.

        """              
        pass

    def store_fit(self):
        """
        Store fit solutions in the 'data/fits/' directory.

        Returns:
            - Tuple: A tuple containing the current timestamp, deposit data, and trace data.

        """
        if not os.path.exists(self.ppConfig.ppData.ROOT + "data/fits/"):
            os.makedirs(self.ppConfig.ppData.ROOT + "data/fits/")
        current_stamp = Logger.stamp()
        Logger.logToStdOut("info", 'Storing solution data in data/fits/')
        deposit = self.deposit_field.to_numpy()
        np.save(self.ppConfig.ppData.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = self.trace_field.to_numpy()
        np.save(self.ppConfig.ppData.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
        return current_stamp, deposit, trace

    def __init__(self, rng, kernels, ppConfig):
        """
        Initialize an instance of PPInternalData with specified random number generator, kernels, and configuration.

        Args:
            - rng: Random number generator for internal data.
            - kernels: Kernels for internal data initialization.
            - ppConfig: Configuration for internal data.

        """
        pass


class PPSimulation:
    """
    PPSimulation: Simulation Class for PolyPhy Library.

    Methods:
        - __drawGUI__(window, ppConfig): Draw GUI elements for the simulation.
        - __init__(ppInternalData, ppConfig, batch_mode, num_iterations): Initialize an instance of PPSimulation with
          specified internal data, configuration, batch mode, and number of iterations.

    Attributes:
        - (Attributes specific to the simulation)

    """
    def __drawGUI__(self, window, ppConfig):
        pass

    def __init__(self, ppInternalData, ppConfig, batch_mode, num_iterations):
        pass


class PPPostSimulation:
    """
    PPPostSimulation: Post-Simulation Class for PolyPhy Library.

    Methods:
        - __init__(ppInternalData): Initialize an instance of PPPostSimulation with specified internal data.

    Attributes:
        - (Attributes specific to post-simulation processing)

    """
    def __init__(self, ppInternalData):
        pass
