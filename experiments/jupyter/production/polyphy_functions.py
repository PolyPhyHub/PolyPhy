from enum import IntEnum
import time
import math, os
from datetime import datetime
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import taichi as ti
import taichi.math as timath

class TypeAliases:
    FLOAT_CPU = np.float32
    INT_CPU = np.int32
    FLOAT_GPU = ti.f32
    INT_GPU = ti.i32

    VEC2i = ti.types.vector(2, INT_GPU)
    VEC3i = ti.types.vector(3, INT_GPU)
    VEC2f = ti.types.vector(2, FLOAT_GPU)
    VEC3f = ti.types.vector(3, FLOAT_GPU)

    #  TODO: Impletement float 16 and 64
    @staticmethod
    def set_precision(float_precision):
        if float_precision == "float64":
            TypeAliases.FLOAT_CPU = np.float64
            TypeAliases.FLOAT_GPU = ti.f64
        elif float_precision == "float32":
            TypeAliases.FLOAT_CPU = np.float32
            TypeAliases.FLOAT_GPU = ti.f32
        elif float_precision == "float16":
            TypeAliases.FLOAT_CPU = np.float16
            TypeAliases.FLOAT_GPU = ti.f16
        else:
            raise ValueError("Invalid float precision value. Supported values: float64, float32, float16")

class PPVariables:
    ## Simulation-wide constants
    N_DATA_DEFAULT = 1000
    N_AGENTS_DEFAULT = 1000000
    DOMAIN_SIZE_DEFAULT = (100.0, 100.0)
    TRACE_RESOLUTION_MAX = 1400
    DEPOSIT_DOWNSCALING_FACTOR = 1
    STEERING_RATE = 0.5
    MAX_DEPOSIT = 10.0
    DOMAIN_MARGIN = 0.05

    ## Distance sampling distribution for agents
    class EnumDistanceSamplingDistribution(IntEnum):
        CONSTANT = 0
        EXPONENTIAL = 1
        MAXWELL_BOLTZMANN = 2

    ## Directional sampling distribution for agents
    class EnumDirectionalSamplingDistribution(IntEnum):
        DISCRETE = 0
        CONE = 1

    ## Sampling strategy for directional agent mutation
    class EnumDirectionalMutationType(IntEnum):
        DETERMINISTIC = 0
        PROBABILISTIC = 1

    ## Deposit fetching strategy
    class EnumDepositFetchingStrategy(IntEnum):
        NN = 0
        NN_PERTURBED = 1

    ## Handling strategy for agents that leave domain boundary
    class EnumAgentBoundaryHandling(IntEnum):
        WRAP = 0
        REINIT_CENTER = 1
        REINIT_RANDOMLY = 2
    
    ## State flags
    distance_sampling_distribution = EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
    directional_sampling_distribution = EnumDirectionalSamplingDistribution.CONE
    directional_mutation_type = EnumDirectionalMutationType.PROBABILISTIC
    deposit_fetching_strategy = EnumDepositFetchingStrategy.NN_PERTURBED
    agent_boundary_handling = EnumAgentBoundaryHandling.WRAP

    @staticmethod
    def set_value(constant_name, new_value):
        if hasattr(PPVariables, constant_name):
            setattr(PPVariables, constant_name, new_value)
        else:
            raise AttributeError(f"'PPVariables' has no attribute '{constant_name}'")
        
    def __init__(self,ppData):
        self.DATA_TO_AGENTS_RATIO = TypeAliases.FLOAT_CPU(ppData.N_DATA) / TypeAliases.FLOAT_CPU(ppData.N_AGENTS)
        self.DOMAIN_SIZE_MAX = np.max([ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1]])
        self.TRACE_RESOLUTION = TypeAliases.INT_CPU((TypeAliases.FLOAT_CPU(PPVariables.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[0] / self.DOMAIN_SIZE_MAX, TypeAliases.FLOAT_CPU(PPVariables.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[1] / self.DOMAIN_SIZE_MAX))
        self.DEPOSIT_RESOLUTION = (self.TRACE_RESOLUTION[0] // PPVariables.DEPOSIT_DOWNSCALING_FACTOR, self.TRACE_RESOLUTION[1] // PPVariables.DEPOSIT_DOWNSCALING_FACTOR)
        self.VIS_RESOLUTION = self.TRACE_RESOLUTION        
        self.sense_distance = 0.005 * self.DOMAIN_SIZE_MAX
        self.sense_angle = 1.5
        self.step_size = 0.0005 * self.DOMAIN_SIZE_MAX
        self.sampling_exponent = 2.0
        self.deposit_attenuation = 0.9
        self.trace_attenuation = 0.96
        self.data_deposit = 0.1 * PPVariables.MAX_DEPOSIT
        self.agent_deposit = self.data_deposit * self.DATA_TO_AGENTS_RATIO
        self.deposit_vis = 0.1
        self.trace_vis = 1.0


class PPData:
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
        data = np.loadtxt(INPUT_FILE, delimiter=",").astype(TypeAliases.FLOAT_CPU)
        N_DATA = data.shape[0]
        N_AGENTS = PPVariables.N_AGENTS_DEFAULT
        domain_min = (np.min(data[:,0]), np.min(data[:,1]))
        domain_max = (np.max(data[:,0]), np.max(data[:,1]))
        domain_size = np.subtract(domain_max, domain_min)
        DOMAIN_MIN = (domain_min[0] - PPVariables.DOMAIN_MARGIN * domain_size[0], domain_min[1] - PPVariables.DOMAIN_MARGIN * domain_size[1])
        DOMAIN_MAX = (domain_max[0] + PPVariables.DOMAIN_MARGIN * domain_size[0], domain_max[1] + PPVariables.DOMAIN_MARGIN * domain_size[1])
        DOMAIN_SIZE = np.subtract(DOMAIN_MAX, DOMAIN_MIN)
        AVG_WEIGHT = np.mean(data[:,2])
    else:
        N_DATA = PPVariables.N_DATA_DEFAULT
        N_AGENTS = PPVariables.N_AGENTS_DEFAULT
        DOMAIN_SIZE = PPVariables.DOMAIN_SIZE_DEFAULT
        DOMAIN_MIN = (0.0, 0.0)
        DOMAIN_MAX = PPVariables.DOMAIN_SIZE_DEFAULT
        data = np.zeros(shape=(N_DATA, 3), dtype = TypeAliases.FLOAT_CPU)
        data[:, 0] = self.rng.normal(loc = DOMAIN_MIN[0] + 0.5 * DOMAIN_MAX[0], scale = 0.13 * DOMAIN_SIZE[0], size = N_DATA)
        data[:, 1] = self.rng.normal(loc = DOMAIN_MIN[1] + 0.5 * DOMAIN_MAX[1], scale = 0.13 * DOMAIN_SIZE[1], size = N_DATA)
        data[:, 2] = AVG_WEIGHT
    
    def __init__(self, rng=default_rng()):
        self.rng = rng  


class Agents:
    ## Init agents
    def __init__(self,rng,ppVariables,ppData):
        self.rng = rng
        self.agents = np.zeros(shape=(ppData.N_AGENTS, 4), dtype = TypeAliases.FLOAT_CPU)
        self.agents[:, 0] = self.rng.uniform(low = ppData.DOMAIN_MIN[0] + 0.001, high = ppData.DOMAIN_MAX[0] - 0.001, size = ppData.N_AGENTS)
        self.agents[:, 1] = self.rng.uniform(low = ppData.DOMAIN_MIN[1] + 0.001, high = ppData.DOMAIN_MAX[1] - 0.001, size = ppData.N_AGENTS)
        self.agents[:, 2] = self.rng.uniform(low = 0.0, high = 2.0 * np.pi, size = ppData.N_AGENTS)
        self.agents[:, 3] = 1.0

        print('Simulation domain min:', ppData.DOMAIN_MIN)
        print('Simulation domain max:', ppData.DOMAIN_MAX)
        print('Simulation domain size:', ppData.DOMAIN_SIZE)
        print('Trace grid resolution:', ppVariables.TRACE_RESOLUTION)
        print('Deposit grid resolution:', ppVariables.DEPOSIT_RESOLUTION)
        print('Data sample:', ppData.data[0, :])
        print('Agent sample:', self.agents[0, :])
        print('Number of agents:', ppData.N_AGENTS)
        print('Number of data points:', ppData.N_DATA)

class FieldVariables:
    ## Allocate GPU memory fields
    ## Keep in mind that the dimensions of these fields are important in the subsequent computations;
    ## that means if they change the GPU kernels and the associated handling code must be modified as well
    def __init__(self,ppVariables,ppData):
        self.data_field = ti.Vector.field(n = 3, dtype = TypeAliases.FLOAT_GPU, shape = ppData.N_DATA)
        self.agents_field = ti.Vector.field(n = 4, dtype = TypeAliases.FLOAT_GPU, shape = ppData.N_AGENTS)
        self.deposit_field = ti.Vector.field(n = 2, dtype = TypeAliases.FLOAT_GPU, shape = ppVariables.DEPOSIT_RESOLUTION)
        self.trace_field = ti.Vector.field(n = 1, dtype = TypeAliases.FLOAT_GPU, shape = ppVariables.TRACE_RESOLUTION)
        self.vis_field = ti.Vector.field(n = 3, dtype = TypeAliases.FLOAT_GPU, shape = ppVariables.VIS_RESOLUTION)
        print('Total GPU memory allocated:', TypeAliases .INT_CPU(4 * (\
            self.data_field.shape[0] * 3 + \
            self.agents_field.shape[0] * 4 + \
            self.deposit_field.shape[0] * self.deposit_field.shape[1] * 2 + \
            self.trace_field.shape[0] * self.trace_field.shape[1] * 1 + \
            self.vis_field.shape[0] * self.vis_field.shape[1] * 3 \
            ) / 2 ** 20), 'MB')
        
class SimulationVisuals:
    def initGPU(self,k):
        ## Initialize GPU fields
        self.fieldVariables.data_field.from_numpy(self.ppData.data)
        self.fieldVariables.agents_field.from_numpy(self.agents.agents)
        k.zero_field(self.fieldVariables.deposit_field)
        k.zero_field(self.fieldVariables.trace_field)
        k.zero_field(self.fieldVariables.vis_field)

    ## Insert a new data point, Round-Robin style, and upload to GPU
    ## This can be very costly for many data points! (eg 10^5 or more)
    def edit_data(self,edit_index: TypeAliases.INT_CPU, window: ti.ui.Window) -> TypeAliases.INT_CPU:
        mouse_rel_pos = window.get_cursor_pos()
        mouse_rel_pos = (np.min([np.max([0.001, window.get_cursor_pos()[0]]), 0.999]), np.min([np.max([0.001, window.get_cursor_pos()[1]]), 0.999]))
        mouse_pos = np.add(self.ppData.DOMAIN_MIN, np.multiply(mouse_rel_pos, self.ppData.DOMAIN_SIZE))
        self.ppData.data[edit_index, :] = mouse_pos[0], mouse_pos[1], self.ppData.AVG_WEIGHT
        self.fieldVariables.data_field.from_numpy(self.ppData.data)
        edit_index = (edit_index + 1) % self.ppData.N_DATA
        return edit_index

    ## Current timestamp
    def stamp(self) -> str:
        return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    ## Store current deposit and trace fields
    def store_fit(self):
        if not os.path.exists(self.ppData.ROOT + "data/fits/"):
            os.makedirs(self.ppData.ROOT + "data/fits/")
        current_stamp = self.stamp()
        deposit = self.fieldVariables.deposit_field.to_numpy()
        np.save(self.ppData.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = self.fieldVariables.trace_field.to_numpy()
        np.save(self.ppData.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
        return current_stamp, deposit, trace

    def __init__(self,k,ppVariables, ppData, agents, fieldVariables):
        self.ppVariables = ppVariables
        self.ppData = ppData
        self.agents = agents
        self.fieldVariables = fieldVariables
        
        self.initGPU(k)
        
        self.current_deposit_index = 0
        self.data_edit_index = 0
        self.do_simulate = True
        self.hide_UI = False

class PolyPhyWindow:
    def __init__(self, k, simulationVisuals):
        ## check if file exists
        if os.path.exists("/tmp/flag") == False:
            window = ti.ui.Window('PolyPhy', (simulationVisuals.fieldVariables.vis_field.shape[0], simulationVisuals.fieldVariables.vis_field.shape[1]), show_window = True)
            window.show()
            canvas = window.get_canvas()
            
            ## Main simulation and rendering loop
            while window.running:
                
                do_export = False
                do_screenshot = False
                do_quit = False
            
                ## Handle controls
                if window.get_event(ti.ui.PRESS):
                    if window.event.key == 'e': do_export = True
                    if window.event.key == 's': do_screenshot = True
                    if window.event.key == 'h': hide_UI = not hide_UI
                    if window.event.key in [ti.ui.ESCAPE]: do_quit = True
                    if window.event.key in [ti.ui.LMB]:
                        simulationVisuals.data_edit_index = simulationVisuals.edit_data(simulationVisuals.data_edit_index,window)
                if window.is_pressed(ti.ui.RMB):
                    simulationVisuals.data_edit_index = simulationVisuals.edit_data(simulationVisuals.data_edit_index,window)
                
                if not simulationVisuals.hide_UI:
                    ## Draw main interactive control GUI
                    window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.ppVariables.VIS_RESOLUTION[0]), 0.74 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.ppVariables.VIS_RESOLUTION[1]))
                    window.GUI.text("MCPM parameters:")
                    simulationVisuals.ppVariables.sense_distance = window.GUI.slider_float('Sensing dist', simulationVisuals.ppVariables.sense_distance, 0.1, 0.05 * np.max([simulationVisuals.ppData.DOMAIN_SIZE[0], simulationVisuals.ppData.DOMAIN_SIZE[1]]))
                    simulationVisuals.ppVariables.sense_angle = window.GUI.slider_float('Sensing angle', simulationVisuals.ppVariables.sense_angle, 0.01, 0.5 * np.pi)
                    simulationVisuals.ppVariables.sampling_exponent = window.GUI.slider_float('Sampling expo', simulationVisuals.ppVariables.sampling_exponent, 1.0, 10.0)
                    simulationVisuals.ppVariables.step_size = window.GUI.slider_float('Step size', simulationVisuals.ppVariables.step_size, 0.0, 0.005 * np.max([simulationVisuals.ppData.DOMAIN_SIZE[0], simulationVisuals.ppData.DOMAIN_SIZE[1]]))
                    simulationVisuals.ppVariables.data_deposit = window.GUI.slider_float('Data deposit', simulationVisuals.ppVariables.data_deposit, 0.0, simulationVisuals.ppVariables.MAX_DEPOSIT)
                    simulationVisuals.ppVariables.agent_deposit = window.GUI.slider_float('Agent deposit', simulationVisuals.ppVariables.agent_deposit, 0.0, 10.0 * simulationVisuals.ppVariables.MAX_DEPOSIT * simulationVisuals.ppVariables.DATA_TO_AGENTS_RATIO)
                    simulationVisuals.ppVariables.deposit_attenuation = window.GUI.slider_float('Deposit attn', simulationVisuals.ppVariables.deposit_attenuation, 0.8, 0.999)
                    simulationVisuals.ppVariables.trace_attenuation = window.GUI.slider_float('Trace attn', simulationVisuals.ppVariables.trace_attenuation, 0.8, 0.999)
                    simulationVisuals.ppVariables.deposit_vis = math.pow(10.0, window.GUI.slider_float('Deposit vis', math.log(simulationVisuals.ppVariables.deposit_vis, 10.0), -3.0, 3.0))
                    simulationVisuals.ppVariables.trace_vis = math.pow(10.0, window.GUI.slider_float('Trace vis', math.log(simulationVisuals.ppVariables.trace_vis, 10.0), -3.0, 3.0))
            
                    window.GUI.text("Distance distribution:")
                    if window.GUI.checkbox("Constant", simulationVisuals.ppVariables.distance_sampling_distribution == simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.CONSTANT):
                        simulationVisuals.ppVariables.distance_sampling_distribution = simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.CONSTANT
                    if window.GUI.checkbox("Exponential", simulationVisuals.ppVariables.distance_sampling_distribution == simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.EXPONENTIAL):
                        simulationVisuals.ppVariables.distance_sampling_distribution = simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.EXPONENTIAL
                    if window.GUI.checkbox("Maxwell-Boltzmann", simulationVisuals.ppVariables.distance_sampling_distribution == simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
                        simulationVisuals.ppVariables.distance_sampling_distribution = simulationVisuals.ppVariables.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
            
                    window.GUI.text("Directional distribution:")
                    if window.GUI.checkbox("Discrete", simulationVisuals.ppVariables.directional_sampling_distribution == simulationVisuals.ppVariables.EnumDirectionalSamplingDistribution.DISCRETE):
                        simulationVisuals.ppVariables.directional_sampling_distribution = simulationVisuals.ppVariables.EnumDirectionalSamplingDistribution.DISCRETE
                    if window.GUI.checkbox("Cone", simulationVisuals.ppVariables.directional_sampling_distribution == simulationVisuals.ppVariables.EnumDirectionalSamplingDistribution.CONE):
                        simulationVisuals.ppVariables.directional_sampling_distribution = simulationVisuals.ppVariables.EnumDirectionalSamplingDistribution.CONE
            
                    window.GUI.text("Directional mutation:")
                    if window.GUI.checkbox("Deterministic", simulationVisuals.ppVariables.directional_mutation_type == simulationVisuals.ppVariables.EnumDirectionalMutationType.DETERMINISTIC):
                        simulationVisuals.ppVariables.directional_mutation_type = simulationVisuals.ppVariables.EnumDirectionalMutationType.DETERMINISTIC
                    if window.GUI.checkbox("Stochastic", simulationVisuals.ppVariables.directional_mutation_type == simulationVisuals.ppVariables.EnumDirectionalMutationType.PROBABILISTIC):
                        simulationVisuals.ppVariables.directional_mutation_type = simulationVisuals.ppVariables.EnumDirectionalMutationType.PROBABILISTIC
            
                    window.GUI.text("Deposit fetching:")
                    if window.GUI.checkbox("Nearest neighbor", simulationVisuals.ppVariables.deposit_fetching_strategy == simulationVisuals.ppVariables.EnumDepositFetchingStrategy.NN):
                        simulationVisuals.ppVariables.deposit_fetching_strategy = simulationVisuals.ppVariables.EnumDepositFetchingStrategy.NN
                    if window.GUI.checkbox("Noise-perturbed NN", simulationVisuals.ppVariables.deposit_fetching_strategy == simulationVisuals.ppVariables.EnumDepositFetchingStrategy.NN_PERTURBED):
                        simulationVisuals.ppVariables.deposit_fetching_strategy = simulationVisuals.ppVariables.EnumDepositFetchingStrategy.NN_PERTURBED
            
                    window.GUI.text("Agent boundary handling:")
                    if window.GUI.checkbox("Wrap around", simulationVisuals.ppVariables.agent_boundary_handling == simulationVisuals.ppVariables.EnumAgentBoundaryHandling.WRAP):
                        simulationVisuals.ppVariables.agent_boundary_handling = simulationVisuals.ppVariables.EnumAgentBoundaryHandling.WRAP
                    if window.GUI.checkbox("Reinitialize center", simulationVisuals.ppVariables.agent_boundary_handling == simulationVisuals.ppVariables.EnumAgentBoundaryHandling.REINIT_CENTER):
                        simulationVisuals.ppVariables.agent_boundary_handling = simulationVisuals.ppVariables.EnumAgentBoundaryHandling.REINIT_CENTER
                    if window.GUI.checkbox("Reinitialize randomly", simulationVisuals.ppVariables.agent_boundary_handling == simulationVisuals.ppVariables.EnumAgentBoundaryHandling.REINIT_RANDOMLY):
                        simulationVisuals.ppVariables.agent_boundary_handling = simulationVisuals.ppVariables.EnumAgentBoundaryHandling.REINIT_RANDOMLY
            
                    window.GUI.text("Misc controls:")
                    simulationVisuals.do_simulate = window.GUI.checkbox("Run simulation", simulationVisuals.do_simulate)
                    do_export = do_export | window.GUI.button('Export fit')
                    do_screenshot = do_screenshot | window.GUI.button('Screenshot')
                    do_quit = do_quit | window.GUI.button('Quit')
                    window.GUI.end()
            
                    ## Help window
                    ## Do not exceed prescribed line length of 120 characters, there is no text wrapping in Taichi GUI for now
                    window.GUI.begin('Help', 0.35 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.ppVariables.VIS_RESOLUTION[0]), 0.01, 0.6, 0.30 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.ppVariables.VIS_RESOLUTION[1]))
                    window.GUI.text("Welcome to PolyPhy 2D GUI variant written by researchers at UCSC/OSPO with the help of numerous external contributors\n(https://github.com/PolyPhyHub). PolyPhy implements MCPM, an agent-based, stochastic, pattern forming algorithm designed\nby Elek et al, inspired by Physarum polycephalum slime mold. Below is a quick reference guide explaining the parameters\nand features available in the interface. The reference as well as other panels can be hidden using the arrow button, moved,\nand rescaled.")
                    window.GUI.text("")
                    window.GUI.text("PARAMETERS")
                    window.GUI.text("Sensing dist: average distance in world units at which agents probe the deposit")
                    window.GUI.text("Sensing angle: angle in radians within which agents probe deposit (left and right concentric to movement direction)")
                    window.GUI.text("Sampling expo: sampling sharpness (or 'acuteness' or 'temperature') which tunes the directional mutation behavior")
                    window.GUI.text("Step size: average size of the step in world units which agents make in each iteration")
                    window.GUI.text("Data deposit: amount of marker 'deposit' that *data* emit at every iteration")
                    window.GUI.text("Agent deposit: amount of marker 'deposit' that *agents* emit at every iteration")
                    window.GUI.text("Deposit attn: attenuation (or 'decay') rate of the diffusing combined agent+data deposit field")
                    window.GUI.text("Trace attn: attenuation (or 'decay') of the non-diffusing agent trace field")
                    window.GUI.text("Deposit vis: visualization intensity of the green deposit field (logarithmic)")
                    window.GUI.text("Trace vis: visualization intensity of the red trace field (logarithmic)")
                    window.GUI.text("")
                    window.GUI.text("OPTIONS")
                    window.GUI.text("Distance distribution: strategy for sampling the sensing and movement distances")
                    window.GUI.text("Directional distribution: strategy for sampling the sensing and movement directions")
                    window.GUI.text("Directional mutation: strategy for selecting the new movement direction")
                    window.GUI.text("Deposit fetching: access behavior when sampling the deposit field")
                    window.GUI.text("Agent boundary handling: what do agents do if they reach the boundary of the simulation domain")
                    window.GUI.text("")
                    window.GUI.text("VISUALIZATION")
                    window.GUI.text("Renders 2 types of information superimposed on top of each other: *green* deposit field and *red-purple* trace field.")
                    window.GUI.text("Yellow-white signifies areas where deposit and trace overlap (relative intensities are controlled by the T/D vis params)")
                    window.GUI.text("Screenshots can be saved in the /capture folder.")
                    window.GUI.text("")
                    window.GUI.text("DATA")
                    window.GUI.text("Input data are loaded from the specified folder in /data. Currently the CSV format is supported.")
                    window.GUI.text("Reconstruction data are exported to /data/fits using the Export fit button.")
                    window.GUI.text("")
                    window.GUI.text("EDITING")
                    window.GUI.text("New data points can be placed by mouse clicking. This overrides old data on a Round-Robin basis.")
                    window.GUI.text("Left mouse: discrete mode, place a single data point")
                    window.GUI.text("Right mouse: continuous mode, place a data point at every iteration")
                    window.GUI.end()
            
                ## Main simulation sequence
                if simulationVisuals.do_simulate:
                    k.data_step(simulationVisuals.fieldVariables.data_field, simulationVisuals.fieldVariables.deposit_field,simulationVisuals.ppVariables.data_deposit, simulationVisuals.current_deposit_index, simulationVisuals.ppData.DOMAIN_MIN, simulationVisuals.ppData.DOMAIN_MAX, simulationVisuals.ppVariables.DEPOSIT_RESOLUTION)
                    k.agent_step(simulationVisuals.ppVariables.sense_distance,\
                        simulationVisuals.ppVariables.sense_angle,\
                        simulationVisuals.ppVariables.STEERING_RATE,\
                        simulationVisuals.ppVariables.sampling_exponent,\
                        simulationVisuals.ppVariables.step_size,\
                        simulationVisuals.ppVariables.agent_deposit,\
                        simulationVisuals.current_deposit_index,\
                        simulationVisuals.ppVariables.distance_sampling_distribution,\
                        simulationVisuals.ppVariables.directional_sampling_distribution,\
                        simulationVisuals.ppVariables.directional_mutation_type,\
                        simulationVisuals.ppVariables.deposit_fetching_strategy,\
                        simulationVisuals.ppVariables.agent_boundary_handling,\
                        simulationVisuals.fieldVariables.agents_field,\
                        simulationVisuals.fieldVariables.deposit_field,\
                        simulationVisuals.fieldVariables.trace_field,\
                        simulationVisuals.ppData.N_DATA,\
                        simulationVisuals.ppData.N_AGENTS,\
                        simulationVisuals.ppData.DOMAIN_SIZE,\
                        simulationVisuals.ppData.DOMAIN_MIN,\
                        simulationVisuals.ppData.DOMAIN_MAX,\
                        simulationVisuals.ppVariables.DEPOSIT_RESOLUTION,\
                        simulationVisuals.ppVariables.TRACE_RESOLUTION
                        )
                    k.deposit_relaxation_step(simulationVisuals.ppVariables.deposit_attenuation, simulationVisuals.current_deposit_index,simulationVisuals.fieldVariables.deposit_field,simulationVisuals.ppVariables.DEPOSIT_RESOLUTION)
                    k.trace_relaxation_step(simulationVisuals.ppVariables.trace_attenuation, simulationVisuals.fieldVariables.trace_field)
                    simulationVisuals.current_deposit_index = 1 - simulationVisuals.current_deposit_index
            
                ## Render visualization
                k.render_visualization(simulationVisuals.ppVariables.deposit_vis, simulationVisuals.ppVariables.trace_vis, simulationVisuals.current_deposit_index, simulationVisuals.fieldVariables.deposit_field,simulationVisuals.fieldVariables.trace_field, simulationVisuals.fieldVariables.vis_field, simulationVisuals.ppVariables.DEPOSIT_RESOLUTION, simulationVisuals.ppVariables.VIS_RESOLUTION, simulationVisuals.ppVariables.TRACE_RESOLUTION)
                canvas.set_image(simulationVisuals.fieldVariables.vis_field)
            
                if do_screenshot:
                    window.save_image(simulationVisuals.ppData.ROOT + 'capture/screenshot_' + simulationVisuals.stamp() + '.png') ## Must appear before window.show() call
                window.show()
                if do_export:
                    simulationVisuals.store_fit()
                if do_quit:
                    break
                
            window.destroy()

class PostSimulation:
    def __init__(self, simulationVisuals):
        ## Store fits
        current_stamp = simulationVisuals.stamp()
        deposit = simulationVisuals.fieldVariables.deposit_field.to_numpy()
        np.save(simulationVisuals.ppData.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = simulationVisuals.fieldVariables.trace_field.to_numpy()
        np.save(simulationVisuals.ppData.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)

        ## Plot results
        ## Compare with stored fields
        current_stamp, deposit, trace = simulationVisuals.store_fit()

        plt.figure(figsize = (10.0, 10.0))
        plt.imshow(np.flip(np.transpose(deposit[:,:,0]), axis=0))
        plt.figure(figsize = (10.0, 10.0))
        deposit_restored = np.load(simulationVisuals.ppData.ROOT + 'data/fits/deposit_' + current_stamp + '.npy')
        plt.imshow(np.flip(np.transpose(deposit_restored[:,:,0]), axis=0))

        plt.figure(figsize = (10.0, 10.0))
        plt.imshow(np.flip(np.transpose(trace[:,:,0]), axis=0))
        plt.figure(figsize = (10.0, 10.0))
        trace_restored = np.load(simulationVisuals.ppData.ROOT + 'data/fits/trace_' + current_stamp + '.npy')
        plt.imshow(np.flip(np.transpose(trace_restored[:,:,0]), axis=0))
