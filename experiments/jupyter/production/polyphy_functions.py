from enum import IntEnum
import time
import math, os
from datetime import datetime
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import taichi as ti
import taichi.math as timath
import zope.interface
import logging

 ## TODO: implement a logging method (instead of print statements) and let user decide whether to log to file or terminal
class PPTypes:
    ## TODO (low priority): impletement and test float 16 and 64

    FLOAT_CPU = np.float32
    INT_CPU = np.int32
    FLOAT_GPU = ti.f32
    INT_GPU = ti.i32

    VEC2i = ti.types.vector(2, INT_GPU)
    VEC3i = ti.types.vector(3, INT_GPU)
    VEC2f = ti.types.vector(2, FLOAT_GPU)
    VEC3f = ti.types.vector(3, FLOAT_GPU)

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
            raise ValueError("Invalid float precision value. Supported values: float64, float32, float16")

class PPConfig:
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

    ## Simulation-wide constants and defaults
    N_DATA_DEFAULT = 1000
    N_AGENTS_DEFAULT = 1000000
    DOMAIN_SIZE_DEFAULT = (100.0, 100.0)
    TRACE_RESOLUTION_MAX = 512
    DEPOSIT_DOWNSCALING_FACTOR = 1
    MAX_DEPOSIT = 10.0
    DOMAIN_MARGIN = 0.05

    @staticmethod
    def set_value(constant_name, new_value):
        CONSTANT_VARS = ["N_DATA_DEFAULT","N_AGENTS_DEFAULT","DOMAIN_SIZE_DEFAULT"]
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

    def __init__(self,ppData):
        self.DATA_TO_AGENTS_RATIO = PPTypes.FLOAT_CPU(ppData.N_DATA) / PPTypes.FLOAT_CPU(ppData.N_AGENTS)
        self.DOMAIN_SIZE_MAX = np.max([ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1]])
        self.TRACE_RESOLUTION = PPTypes.INT_CPU((PPTypes.FLOAT_CPU(PPConfig.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[0] / self.DOMAIN_SIZE_MAX, PPTypes.FLOAT_CPU(PPConfig.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[1] / self.DOMAIN_SIZE_MAX))
        self.DEPOSIT_RESOLUTION = (self.TRACE_RESOLUTION[0] // PPConfig.DEPOSIT_DOWNSCALING_FACTOR, self.TRACE_RESOLUTION[1] // PPConfig.DEPOSIT_DOWNSCALING_FACTOR)
        self.VIS_RESOLUTION = self.TRACE_RESOLUTION
        self.sense_distance = 0.005 * self.DOMAIN_SIZE_MAX
        self.sense_angle = 1.5
        self.steering_rate = 0.5
        self.step_size = 0.0005 * self.DOMAIN_SIZE_MAX
        self.sampling_exponent = 2.0
        self.deposit_attenuation = 0.9
        self.trace_attenuation = 0.96
        self.data_deposit = 0.1 * PPConfig.MAX_DEPOSIT
        self.agent_deposit = self.data_deposit * self.DATA_TO_AGENTS_RATIO
        self.deposit_vis = 0.1
        self.trace_vis = 1.0
        PPUtils.logToStdOut('info','Trace grid resolution:', self.TRACE_RESOLUTION)
        PPUtils.logToStdOut('info','Deposit grid resolution:', self.DEPOSIT_RESOLUTION)

class PPInputData(zope.interface.Interface):
    def load_from_file(file):
        # Load from a file - parse file extension
        pass

    def generate_test_data(rng):
        # Load random data to test / simulation
        pass

@zope.interface.implementer(PPInputData)
class PPInputData_2DDiscrete:
    ## TODO: determine ROOT automatically
    ROOT = '../../../'
    def load_from_file(self,file = 'data/csv/sample_2D_linW.csv'):
        ## TODO: implement file loader for different file types
        self.data = np.loadtxt('../../../' + file, delimiter=",").astype(PPTypes.FLOAT_CPU)
        self.N_DATA = self.data.shape[0]
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.domain_min = (np.min(self.data[:,0]), np.min(self.data[:,1]))
        self.domain_max = (np.max(self.data[:,0]), np.max(self.data[:,1]))
        self.domain_size = np.subtract(self.domain_max, self.domain_min)
        self.DOMAIN_MIN = (self.domain_min[0] - PPConfig.DOMAIN_MARGIN * self.domain_size[0], self.domain_min[1] - PPConfig.DOMAIN_MARGIN * self.domain_size[1])
        self.DOMAIN_MAX = (self.domain_max[0] + PPConfig.DOMAIN_MARGIN * self.domain_size[0], self.domain_max[1] + PPConfig.DOMAIN_MARGIN * self.domain_size[1])
        self.DOMAIN_SIZE = np.subtract(self.DOMAIN_MAX, self.DOMAIN_MIN)
        self.AVG_WEIGHT = np.mean(self.data[:,2])

    def generate_test_data(self,rng):
        self.N_DATA = PPConfig.N_DATA_DEFAULT
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.DOMAIN_SIZE = PPConfig.DOMAIN_SIZE_DEFAULT
        self.DOMAIN_MIN = (0.0, 0.0)
        self.DOMAIN_MAX = PPConfig.DOMAIN_SIZE_DEFAULT
        self.data = np.zeros(shape=(self.N_DATA, 3), dtype = PPTypes.FLOAT_CPU)
        self.data[:, 0] = rng.normal(loc = self.DOMAIN_MIN[0] + 0.5 * self.DOMAIN_MAX[0], scale = 0.13 * self.DOMAIN_SIZE[0], size = self.N_DATA)
        self.data[:, 1] = rng.normal(loc = self.DOMAIN_MIN[1] + 0.5 * self.DOMAIN_MAX[1], scale = 0.13 * self.DOMAIN_SIZE[1], size = self.N_DATA)
        self.data[:, 2] = np.mean(self.data[:,2])
    
    def print_simulation_data(self):
        PPUtils.logToStdOut("info",'Simulation domain min:', self.DOMAIN_MIN)
        PPUtils.logToStdOut("info",'Simulation domain max:', self.DOMAIN_MAX)
        PPUtils.logToStdOut("info",'Simulation domain size:', self.DOMAIN_SIZE)
        PPUtils.logToStdOut("info",'Data sample:', self.data[0, :])
        PPUtils.logToStdOut("info",'Number of agents:', self.N_AGENTS)
        PPUtils.logToStdOut("info",'Number of data points:', self.N_DATA)

    def __init__(self, input_file, rng=default_rng()):
        ## Initialize data and agents
        self.data = None
        self.DOMAIN_MIN = None
        self.DOMAIN_MAX = None
        self.DOMAIN_SIZE = None
        self.N_DATA = None
        self.N_AGENTS = None
        self.AVG_WEIGHT = 10.0
        if len(input_file) > 0:
            self.load_from_file(input_file)
        else:
            self.generate_test_data(rng)
            ## TODO: load data from specified file + type
        self.print_simulation_data()

class PPInternalData:    
    def initInternalData(self,kernels):
        ## Initialize GPU fields
        self.data_field.from_numpy(self.ppData.data)
        self.agents_field.from_numpy(self.agents)
        kernels.zero_field(self.deposit_field)
        kernels.zero_field(self.trace_field)
        kernels.zero_field(self.vis_field)

    ## Insert a new data point, Round-Robin style, and upload to GPU
    ## This can be very costly for many data points! (eg 10^5 or more)
    def edit_data(self,edit_index: PPTypes.INT_CPU, window: ti.ui.Window) -> PPTypes.INT_CPU:
        mouse_rel_pos = (np.min([np.max([0.001, window.get_cursor_pos()[0]]), 0.999]), np.min([np.max([0.001, window.get_cursor_pos()[1]]), 0.999]))
        mouse_pos = np.add(self.ppData.DOMAIN_MIN, np.multiply(mouse_rel_pos, self.ppData.DOMAIN_SIZE))
        self.ppData.data[edit_index, :] = mouse_pos[0], mouse_pos[1], self.ppData.AVG_WEIGHT
        self.data_field.from_numpy(self.ppData.data)
        edit_index = (edit_index + 1) % self.ppData.N_DATA
        return edit_index
    
    ## Store current deposit and trace fields
    def store_fit(self):
        if not os.path.exists(self.ppData.ROOT + "data/fits/"):
            os.makedirs(self.ppData.ROOT + "data/fits/")
        current_stamp = PPUtils.stamp()
        deposit = self.deposit_field.to_numpy()
        np.save(self.ppData.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = self.trace_field.to_numpy()
        np.save(self.ppData.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
        return current_stamp, deposit, trace
    
    def __init__(self,rng,kernels,ppConfig,ppData):
        self.agents = np.zeros(shape=(ppData.N_AGENTS, 4), dtype = PPTypes.FLOAT_CPU)
        self.agents[:, 0] = rng.uniform(low = ppData.DOMAIN_MIN[0] + 0.001, high = ppData.DOMAIN_MAX[0] - 0.001, size = ppData.N_AGENTS)
        self.agents[:, 1] = rng.uniform(low = ppData.DOMAIN_MIN[1] + 0.001, high = ppData.DOMAIN_MAX[1] - 0.001, size = ppData.N_AGENTS)
        self.agents[:, 2] = rng.uniform(low = 0.0, high = 2.0 * np.pi, size = ppData.N_AGENTS)
        self.agents[:, 3] = 1.0
        PPUtils.logToStdOut("info",'Agent sample:', self.agents[0, :])

        self.data_field = ti.Vector.field(n = 3, dtype = PPTypes.FLOAT_GPU, shape = ppData.N_DATA)
        self.agents_field = ti.Vector.field(n = 4, dtype = PPTypes.FLOAT_GPU, shape = ppData.N_AGENTS)
        self.deposit_field = ti.Vector.field(n = 2, dtype = PPTypes.FLOAT_GPU, shape = ppConfig.DEPOSIT_RESOLUTION)
        self.trace_field = ti.Vector.field(n = 1, dtype = PPTypes.FLOAT_GPU, shape = ppConfig.TRACE_RESOLUTION)
        self.vis_field = ti.Vector.field(n = 3, dtype = PPTypes.FLOAT_GPU, shape = ppConfig.VIS_RESOLUTION)
        PPUtils.logToFile("info",'Total GPU memory allocated:', PPTypes.INT_CPU(4 * (\
            self.data_field.shape[0] * 3 + \
            self.agents_field.shape[0] * 4 + \
            self.deposit_field.shape[0] * self.deposit_field.shape[1] * 2 + \
            self.trace_field.shape[0] * self.trace_field.shape[1] * 1 + \
            self.vis_field.shape[0] * self.vis_field.shape[1] * 3 \
            ) / 2 ** 20), 'MB')
        
        self.ppData = ppData
        self.initInternalData(kernels)

class PPUtils:
    log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    @staticmethod
    def stamp() -> str:
        return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    file_logger = logging.getLogger("file")
    file_handler = logging.FileHandler("polyphy.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    file_log_functions = {
        'debug': file_logger.debug,
        'info': file_logger.info,
        'warning': file_logger.warning,
        'error': file_logger.error,
        'critical': file_logger.critical
    }

    @staticmethod
    def logToFile(level, *msg) -> None:
        PPUtils.file_logger.setLevel(PPUtils.log_level.get(level, logging.INFO))
        res = " ".join(map(str, msg))
        PPUtils.file_log_functions.get(level, PPUtils.file_logger.info)(res)
    
    console_logger = logging.getLogger("console")
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    console_log_functions = {
        'debug': console_logger.debug,
        'info': console_logger.info,
        'warning': console_logger.warning,
        'error': console_logger.error,
        'critical': console_logger.critical
    }

    @staticmethod
    def logToStdOut(level, *msg) -> None:
        PPUtils.console_logger.setLevel(PPUtils.log_level.get(level, logging.INFO))
        res = " ".join(map(str, msg))
        PPUtils.previous_log_message = res
        PPUtils.console_log_functions.get(level, PPUtils.console_logger.info)(res)

class PPSimulation:
    def __drawGUI__(self,window,ppConfig,ppData):
        ## Draw main interactive control GUI
        window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[0]), 0.74 * 1024.0 / PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[1]))
        window.GUI.text("MCPM parameters:")
        ppConfig.sense_distance = window.GUI.slider_float('Sensing dist', ppConfig.sense_distance, 0.1, 0.05 * np.max([ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1]]))
        ppConfig.sense_angle = window.GUI.slider_float('Sensing angle', ppConfig.sense_angle, 0.01, 0.5 * np.pi)
        ppConfig.sampling_exponent = window.GUI.slider_float('Sampling expo', ppConfig.sampling_exponent, 1.0, 10.0)
        ppConfig.step_size = window.GUI.slider_float('Step size', ppConfig.step_size, 0.0, 0.005 * np.max([ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1]]))
        ppConfig.data_deposit = window.GUI.slider_float('Data deposit', ppConfig.data_deposit, 0.0, ppConfig.MAX_DEPOSIT)
        ppConfig.agent_deposit = window.GUI.slider_float('Agent deposit', ppConfig.agent_deposit, 0.0, 10.0 * ppConfig.MAX_DEPOSIT * ppConfig.DATA_TO_AGENTS_RATIO)
        ppConfig.deposit_attenuation = window.GUI.slider_float('Deposit attn', ppConfig.deposit_attenuation, 0.8, 0.999)
        ppConfig.trace_attenuation = window.GUI.slider_float('Trace attn', ppConfig.trace_attenuation, 0.8, 0.999)
        ppConfig.deposit_vis = math.pow(10.0, window.GUI.slider_float('Deposit vis', math.log(ppConfig.deposit_vis, 10.0), -3.0, 3.0))
        ppConfig.trace_vis = math.pow(10.0, window.GUI.slider_float('Trace vis', math.log(ppConfig.trace_vis, 10.0), -3.0, 3.0))

        window.GUI.text("Distance distribution:")
        if window.GUI.checkbox("Constant", ppConfig.distance_sampling_distribution == ppConfig.EnumDistanceSamplingDistribution.CONSTANT):
            ppConfig.distance_sampling_distribution = ppConfig.EnumDistanceSamplingDistribution.CONSTANT
        if window.GUI.checkbox("Exponential", ppConfig.distance_sampling_distribution == ppConfig.EnumDistanceSamplingDistribution.EXPONENTIAL):
            ppConfig.distance_sampling_distribution = ppConfig.EnumDistanceSamplingDistribution.EXPONENTIAL
        if window.GUI.checkbox("Maxwell-Boltzmann", ppConfig.distance_sampling_distribution == ppConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
            ppConfig.distance_sampling_distribution = ppConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN

        window.GUI.text("Directional distribution:")
        if window.GUI.checkbox("Discrete", ppConfig.directional_sampling_distribution == ppConfig.EnumDirectionalSamplingDistribution.DISCRETE):
            ppConfig.directional_sampling_distribution = ppConfig.EnumDirectionalSamplingDistribution.DISCRETE
        if window.GUI.checkbox("Cone", ppConfig.directional_sampling_distribution == ppConfig.EnumDirectionalSamplingDistribution.CONE):
            ppConfig.directional_sampling_distribution = ppConfig.EnumDirectionalSamplingDistribution.CONE

        window.GUI.text("Directional mutation:")
        if window.GUI.checkbox("Deterministic", ppConfig.directional_mutation_type == ppConfig.EnumDirectionalMutationType.DETERMINISTIC):
            ppConfig.directional_mutation_type = ppConfig.EnumDirectionalMutationType.DETERMINISTIC
        if window.GUI.checkbox("Stochastic", ppConfig.directional_mutation_type == ppConfig.EnumDirectionalMutationType.PROBABILISTIC):
            ppConfig.directional_mutation_type = ppConfig.EnumDirectionalMutationType.PROBABILISTIC

        window.GUI.text("Deposit fetching:")
        if window.GUI.checkbox("Nearest neighbor", ppConfig.deposit_fetching_strategy == ppConfig.EnumDepositFetchingStrategy.NN):
            ppConfig.deposit_fetching_strategy = ppConfig.EnumDepositFetchingStrategy.NN
        if window.GUI.checkbox("Noise-perturbed NN", ppConfig.deposit_fetching_strategy == ppConfig.EnumDepositFetchingStrategy.NN_PERTURBED):
            ppConfig.deposit_fetching_strategy = ppConfig.EnumDepositFetchingStrategy.NN_PERTURBED

        window.GUI.text("Agent boundary handling:")
        if window.GUI.checkbox("Wrap around", ppConfig.agent_boundary_handling == ppConfig.EnumAgentBoundaryHandling.WRAP):
            ppConfig.agent_boundary_handling = ppConfig.EnumAgentBoundaryHandling.WRAP
        if window.GUI.checkbox("Reinitialize center", ppConfig.agent_boundary_handling == ppConfig.EnumAgentBoundaryHandling.REINIT_CENTER):
            ppConfig.agent_boundary_handling = ppConfig.EnumAgentBoundaryHandling.REINIT_CENTER
        if window.GUI.checkbox("Reinitialize randomly", ppConfig.agent_boundary_handling == ppConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY):
            ppConfig.agent_boundary_handling = ppConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY

        window.GUI.text("Misc controls:")
        self.do_simulate = window.GUI.checkbox("Run simulation", self.do_simulate)
        self.do_export = self.do_export | window.GUI.button('Export fit')
        self.do_screenshot = self.do_screenshot | window.GUI.button('Screenshot')
        self.do_quit = self.do_quit | window.GUI.button('Quit')
        window.GUI.end()

        ## Help window
        ## Do not exceed prescribed line length of 120 characters, there is no text wrapping in Taichi GUI for now
        window.GUI.begin('Help', 0.35 * 1024.0 / PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[0]), 0.01, 0.6, 0.30 * 1024.0 / PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[1]))
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

    def __init__(self, kernels, ppInternalData, ppConfig, ppData, batch_mode=False, num_iterations=-1):
        self.current_deposit_index = 0
        self.data_edit_index = 0

        self.do_export = False
        self.do_screenshot = False
        self.do_quit = False
        self.do_simulate = True
        self.hide_UI = False

        ## check if file exists
        if os.path.exists("/tmp/flag") == False:
            if batch_mode is False:
                window = ti.ui.Window('PolyPhy', (ppInternalData.vis_field.shape[0], ppInternalData.vis_field.shape[1]), show_window = True)
                window.show()
                canvas = window.get_canvas()
            
            curr_iteration = 0
            ## Main simulation and rendering loop
            while window.running if 'window' in locals() else True:
                if batch_mode is True and curr_iteration > num_iterations:
                    break
                curr_iteration = curr_iteration+1

                if batch_mode is False:
                    ## Handle controls
                    if window.get_event(ti.ui.PRESS):
                        if window.event.key == 'e': self.do_export = True
                        if window.event.key == 's': self.do_screenshot = True
                        if window.event.key == 'h': self.hide_UI = not self.hide_UI
                        if window.event.key in [ti.ui.ESCAPE]: self.do_quit = True
                        if window.event.key in [ti.ui.LMB]:
                            self.data_edit_index = ppInternalData.edit_data(self.data_edit_index,window)
                    if window.is_pressed(ti.ui.RMB):
                        self.data_edit_index = ppInternalData.edit_data(self.data_edit_index,window)
                
                    if not self.hide_UI:
                        self.__drawGUI__(window,ppConfig,ppData)
            
                ## Main simulation sequence
                if self.do_simulate:
                    kernels.data_step(ppInternalData.data_field, ppInternalData.deposit_field,ppConfig.data_deposit, self.current_deposit_index, ppData.DOMAIN_MIN, ppData.DOMAIN_MAX, ppConfig.DEPOSIT_RESOLUTION)
                    kernels.agent_step(ppConfig.sense_distance,\
                        ppConfig.sense_angle,\
                        ppConfig.steering_rate,\
                        ppConfig.sampling_exponent,\
                        ppConfig.step_size,\
                        ppConfig.agent_deposit,\
                        self.current_deposit_index,\
                        ppConfig.distance_sampling_distribution,\
                        ppConfig.directional_sampling_distribution,\
                        ppConfig.directional_mutation_type,\
                        ppConfig.deposit_fetching_strategy,\
                        ppConfig.agent_boundary_handling,\
                        ppData.N_DATA,\
                        ppData.N_AGENTS,\
                        ppData.DOMAIN_SIZE,\
                        ppData.DOMAIN_MIN,\
                        ppData.DOMAIN_MAX,\
                        ppConfig.DEPOSIT_RESOLUTION,\
                        ppConfig.TRACE_RESOLUTION,\
                        ppInternalData.agents_field,\
                        ppInternalData.deposit_field,\
                        ppInternalData.trace_field
                        )
                    kernels.deposit_relaxation_step(ppConfig.deposit_attenuation, self.current_deposit_index,ppConfig.DEPOSIT_RESOLUTION,ppInternalData.deposit_field)
                    kernels.trace_relaxation_step(ppConfig.trace_attenuation, ppInternalData.trace_field)
                    self.current_deposit_index = 1 - self.current_deposit_index
            
                ## Render visualization
                kernels.render_visualization(ppConfig.deposit_vis, ppConfig.trace_vis, self.current_deposit_index, ppConfig.DEPOSIT_RESOLUTION, ppConfig.VIS_RESOLUTION, ppConfig.TRACE_RESOLUTION, ppInternalData.deposit_field,ppInternalData.trace_field, ppInternalData.vis_field)
                
                if batch_mode is False:
                    canvas.set_image(ppInternalData.vis_field)
                    if self.do_screenshot:
                        window.save_image(ppData.ROOT + 'capture/screenshot_' + PPUtils.stamp() + '.png') ## Must appear before window.show() call
                    window.show()
                if self.do_export:
                    ppInternalData.store_fit()
                if self.do_quit:
                    break
            if batch_mode is False:    
                window.destroy()

class PPPostSimulation:
    def __init__(self, ppInternalData):
        ppInternalData.store_fit()

