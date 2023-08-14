from first import TypeAliases 
from third import SimulationConstants
import taichi as ti
from numpy.random import default_rng
import numpy as np

## Initialize Taichi
ti.init(arch=ti.cpu)
rng = default_rng()

class DataLoader:
    ## Default root directory
    ROOT = '../../../'

    ## Data input file - leave empty for random set
    INPUT_FILE = ROOT + 'data/csv/sample_2D_linW.csv'

    ## Initialize data and agents
    data = None
    DOMAIN_MIN = None
    DOMAIN_MAX = None
    DOMAIN_SIZE = None
    N_DATA = None
    N_AGENTS = None
    AVG_WEIGHT = 10.0

    ## Load data
    ## If no input file then generate a random dataset
    if len(INPUT_FILE) > 0:
        data = np.loadtxt(INPUT_FILE, delimiter=",").astype(TypeAliases .FLOAT_CPU)
        N_DATA = data.shape[0]
        N_AGENTS = SimulationConstants.N_AGENTS_DEFAULT
        domain_min = (np.min(data[:,0]), np.min(data[:,1]))
        domain_max = (np.max(data[:,0]), np.max(data[:,1]))
        domain_size = np.subtract(domain_max, domain_min)
        DOMAIN_MIN = (domain_min[0] - SimulationConstants.DOMAIN_MARGIN * domain_size[0], domain_min[1] - SimulationConstants.DOMAIN_MARGIN * domain_size[1])
        DOMAIN_MAX = (domain_max[0] + SimulationConstants.DOMAIN_MARGIN * domain_size[0], domain_max[1] + SimulationConstants.DOMAIN_MARGIN * domain_size[1])
        DOMAIN_SIZE = np.subtract(DOMAIN_MAX, DOMAIN_MIN)
        AVG_WEIGHT = np.mean(data[:,2])
    else:
        N_DATA = SimulationConstants.N_DATA_DEFAULT
        N_AGENTS = SimulationConstants.N_AGENTS_DEFAULT
        DOMAIN_SIZE = SimulationConstants.DOMAIN_SIZE_DEFAULT
        DOMAIN_MIN = (0.0, 0.0)
        DOMAIN_MAX = SimulationConstants.DOMAIN_SIZE_DEFAULT
        data = np.zeros(shape=(N_DATA, 3), dtype = TypeAliases.FLOAT_CPU)
        data[:, 0] = rng.normal(loc = DOMAIN_MIN[0] + 0.5 * DOMAIN_MAX[0], scale = 0.13 * DOMAIN_SIZE[0], size = N_DATA)
        data[:, 1] = rng.normal(loc = DOMAIN_MIN[1] + 0.5 * DOMAIN_MAX[1], scale = 0.13 * DOMAIN_SIZE[1], size = N_DATA)
        data[:, 2] = AVG_WEIGHT

class DerivedVariables:
    ## Derived constants
    def __init__(self,dataLoader=DataLoader()):
        self.DATA_TO_AGENTS_RATIO = TypeAliases.FLOAT_CPU(dataLoader.N_DATA) / TypeAliases.FLOAT_CPU(dataLoader.N_AGENTS)
        self.DOMAIN_SIZE_MAX = np.max([dataLoader.DOMAIN_SIZE[0], dataLoader.DOMAIN_SIZE[1]])
        self.TRACE_RESOLUTION = TypeAliases.INT_CPU((TypeAliases.FLOAT_CPU(SimulationConstants.TRACE_RESOLUTION_MAX) * dataLoader.DOMAIN_SIZE[0] / self.DOMAIN_SIZE_MAX, TypeAliases.FLOAT_CPU(SimulationConstants.TRACE_RESOLUTION_MAX) * dataLoader.DOMAIN_SIZE[1] / self.DOMAIN_SIZE_MAX))
        self.DEPOSIT_RESOLUTION = (self.TRACE_RESOLUTION[0] // SimulationConstants.DEPOSIT_DOWNSCALING_FACTOR, self.TRACE_RESOLUTION[1] // SimulationConstants.DEPOSIT_DOWNSCALING_FACTOR)
        self.VIS_RESOLUTION = self.TRACE_RESOLUTION

class Agents:
    ## Init agents
    def __init__(self,dataLoader=DataLoader(),derivedVariables=DerivedVariables()):
        self.agents = np.zeros(shape=(dataLoader.N_AGENTS, 4), dtype = TypeAliases.FLOAT_CPU)
        self.agents[:, 0] = rng.uniform(low = dataLoader.DOMAIN_MIN[0] + 0.001, high = dataLoader.DOMAIN_MAX[0] - 0.001, size = dataLoader.N_AGENTS)
        self.agents[:, 1] = rng.uniform(low = dataLoader.DOMAIN_MIN[1] + 0.001, high = dataLoader.DOMAIN_MAX[1] - 0.001, size = dataLoader.N_AGENTS)
        self.agents[:, 2] = rng.uniform(low = 0.0, high = 2.0 * np.pi, size = dataLoader.N_AGENTS)
        self.agents[:, 3] = 1.0

        print('Simulation domain min:', dataLoader.DOMAIN_MIN)
        print('Simulation domain max:', dataLoader.DOMAIN_MAX)
        print('Simulation domain size:', dataLoader.DOMAIN_SIZE)
        print('Trace grid resolution:', derivedVariables.TRACE_RESOLUTION)
        print('Deposit grid resolution:', derivedVariables.DEPOSIT_RESOLUTION)
        print('Data sample:', dataLoader.data[0, :])
        print('Agent sample:', self.agents[0, :])
        print('Number of agents:', dataLoader.N_AGENTS)
        print('Number of data points:', dataLoader.N_DATA)

class FieldVariables:
    ## Allocate GPU memory fields
    ## Keep in mind that the dimensions of these fields are important in the subsequent computations;
    ## that means if they change the GPU kernels and the associated handling code must be modified as well
    def __init__(self,dataLoader=DataLoader(),derivedVariables=DerivedVariables()):
        self.data_field = ti.Vector.field(n = 3, dtype = TypeAliases.FLOAT_GPU, shape = dataLoader.N_DATA)
        self.agents_field = ti.Vector.field(n = 4, dtype = TypeAliases.FLOAT_GPU, shape = dataLoader.N_AGENTS)
        self.deposit_field = ti.Vector.field(n = 2, dtype = TypeAliases.FLOAT_GPU, shape = derivedVariables.DEPOSIT_RESOLUTION)
        self.trace_field = ti.Vector.field(n = 1, dtype = TypeAliases.FLOAT_GPU, shape = derivedVariables.TRACE_RESOLUTION)
        self.vis_field = ti.Vector.field(n = 3, dtype = TypeAliases.FLOAT_GPU, shape = derivedVariables.VIS_RESOLUTION)
        print('Total GPU memory allocated:', TypeAliases .INT_CPU(4 * (\
            self.data_field.shape[0] * 3 + \
            self.agents_field.shape[0] * 4 + \
            self.deposit_field.shape[0] * self.deposit_field.shape[1] * 2 + \
            self.trace_field.shape[0] * self.trace_field.shape[1] * 1 + \
            self.vis_field.shape[0] * self.vis_field.shape[1] * 3 \
            ) / 2 ** 20), 'MB')