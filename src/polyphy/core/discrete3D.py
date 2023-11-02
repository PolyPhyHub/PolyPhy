import os
import numpy as np
import taichi as ti

from .common import PPTypes, PPConfig, PPInputData, PPInternalData
from .common import PPSimulation, PPPostSimulation
from utils.gui_helper import GuiHelper
from utils.logger import Logger


class PPConfig_3DDiscrete(PPConfig):
    def __init__(self):
        super()

    def register_data(self, ppData):
        self.ppData = ppData
        self.TRACE_RESOLUTION_MAX = 512
        self.DATA_TO_AGENTS_RATIO = PPTypes.FLOAT_CPU(
            ppData.N_DATA) / PPTypes.FLOAT_CPU(ppData.N_AGENTS)
        self.DOMAIN_SIZE_MAX = np.max(
            [ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1], ppData.DOMAIN_SIZE[2]])
        self.TRACE_RESOLUTION = PPTypes.INT_CPU((
            PPTypes.FLOAT_CPU(self.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[0] /
            self.DOMAIN_SIZE_MAX,
            PPTypes.FLOAT_CPU(self.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[1] /
            self.DOMAIN_SIZE_MAX,
            PPTypes.FLOAT_CPU(self.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[2] /
            self.DOMAIN_SIZE_MAX))
        self.DEPOSIT_RESOLUTION = (
            self.TRACE_RESOLUTION[0] // PPConfig.DEPOSIT_DOWNSCALING_FACTOR,
            self.TRACE_RESOLUTION[1] // PPConfig.DEPOSIT_DOWNSCALING_FACTOR,
            self.TRACE_RESOLUTION[2] // PPConfig.DEPOSIT_DOWNSCALING_FACTOR)

        # Check if these are set and if not give them decent initial estimates
        if self.sense_distance < 1.e-4:
            self.sense_distance = 0.005 * self.DOMAIN_SIZE_MAX
        if self.step_size < 1.e-5:
            self.step_size = 0.0005 * self.DOMAIN_SIZE_MAX
        if self.data_deposit < -1.e-4:
            self.data_deposit = 0.1 * PPConfig.MAX_DEPOSIT
        if self.agent_deposit < -1.e-5:
            self.agent_deposit = self.data_deposit * self.DATA_TO_AGENTS_RATIO
        self.input_file = ppData.input_file
        Logger.logToStdOut('info', 'Trace grid resolution:', self.TRACE_RESOLUTION)
        Logger.logToStdOut('info', 'Deposit grid resolution:', self.DEPOSIT_RESOLUTION)
        Logger.logToStdOut('info', 'Vis resolution:', self.VIS_RESOLUTION)


class PPInputData_3DDiscrete(PPInputData):
    # TODO: load datasets from specified file + type

    def __load_from_file__(self):
        Logger.logToStdOut("info", 'Loading input file... '
                           + self.ROOT + self.input_file, self.DOMAIN_MIN)
        self.data = np.loadtxt(
            self.ROOT + self.input_file, delimiter=",").astype(PPTypes.FLOAT_CPU)
        self.N_DATA = self.data.shape[0]
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.domain_min = (
            np.min(self.data[:, 0]),
            np.min(self.data[:, 1]),
            np.min(self.data[:, 2]))
        self.domain_max = (
            np.max(self.data[:, 0]),
            np.max(self.data[:, 1]),
            np.max(self.data[:, 2]))
        self.domain_size = np.subtract(self.domain_max, self.domain_min)
        self.DOMAIN_MIN = (
            self.domain_min[0] - PPConfig.DOMAIN_MARGIN * self.domain_size[0],
            self.domain_min[1] - PPConfig.DOMAIN_MARGIN * self.domain_size[1],
            self.domain_min[2] - PPConfig.DOMAIN_MARGIN * self.domain_size[2])
        self.DOMAIN_MAX = (
            self.domain_max[0] + PPConfig.DOMAIN_MARGIN * self.domain_size[0],
            self.domain_max[1] + PPConfig.DOMAIN_MARGIN * self.domain_size[1],
            self.domain_max[2] + PPConfig.DOMAIN_MARGIN * self.domain_size[2])
        self.DOMAIN_CENTER = (
            0.5 * (self.DOMAIN_MIN[0] + self.DOMAIN_MAX[0]),
            0.5 * (self.DOMAIN_MIN[1] + self.DOMAIN_MAX[1]),
            0.5 * (self.DOMAIN_MIN[2] + self.DOMAIN_MAX[2]))
        self.DOMAIN_SIZE = np.subtract(self.DOMAIN_MAX, self.DOMAIN_MIN)
        self.AVG_WEIGHT = np.mean(self.data[:, 3])

    def __generate_test_data__(self, rng):
        Logger.logToStdOut("info", 'Generating synthetic testing dataset...')
        self.AVG_WEIGHT = 10.0
        self.N_DATA = PPConfig.N_DATA_DEFAULT
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.DOMAIN_SIZE = (
            PPConfig.DOMAIN_SIZE_DEFAULT,
            PPConfig.DOMAIN_SIZE_DEFAULT,
            PPConfig.DOMAIN_SIZE_DEFAULT)
        self.DOMAIN_MIN = (0.0, 0.0, 0.0)
        self.DOMAIN_MAX = (
            PPConfig.DOMAIN_SIZE_DEFAULT,
            PPConfig.DOMAIN_SIZE_DEFAULT,
            PPConfig.DOMAIN_SIZE_DEFAULT)
        self.DOMAIN_CENTER = (
            0.5 * (self.DOMAIN_MIN[0] + self.DOMAIN_MAX[0]),
            0.5 * (self.DOMAIN_MIN[1] + self.DOMAIN_MAX[1]),
            0.5 * (self.DOMAIN_MIN[2] + self.DOMAIN_MAX[2]))
        self.data = np.zeros(
            shape=(self.N_DATA, 4),
            dtype=PPTypes.FLOAT_CPU)
        self.data[:, 0] = rng.normal(
            loc=self.DOMAIN_MIN[0] + 0.5 * self.DOMAIN_MAX[0],
            scale=0.15 * self.DOMAIN_SIZE[0],
            size=self.N_DATA)
        self.data[:, 1] = rng.normal(
            loc=self.DOMAIN_MIN[1] + 0.5 * self.DOMAIN_MAX[1],
            scale=0.15 * self.DOMAIN_SIZE[1],
            size=self.N_DATA)
        self.data[:, 2] = rng.normal(
            loc=self.DOMAIN_MIN[2] + 0.5 * self.DOMAIN_MAX[2],
            scale=0.15 * self.DOMAIN_SIZE[2],
            size=self.N_DATA)
        self.data[:, 3] = self.AVG_WEIGHT


class PPInternalData_3DDiscrete(PPInternalData):
    def __init_internal_data__(self, ppKernels):
        # Initialize GPU fields
        self.data_field.from_numpy(self.ppConfig.ppData.data)
        self.agents_field.from_numpy(self.agents)
        ppKernels.zero_field(self.deposit_field)
        ppKernels.zero_field(self.trace_field)
        ppKernels.zero_field(self.vis_field)

    # Store current deposit and trace fields
    def store_fit(self):
        if not os.path.exists(self.ppConfig.ppData.ROOT + "data/fits/"):
            os.makedirs(self.ppConfig.ppData.ROOT + "data/fits/")
        current_stamp = Logger.stamp()
        Logger.logToStdOut("info", 'Storing solution data in data/fits/')
        deposit = self.deposit_field.to_numpy()
        np.save(
            self.ppConfig.ppData.ROOT + 'data/fits/deposit_' + current_stamp +
            '.npy', deposit)
        trace = self.trace_field.to_numpy()
        np.save(
            self.ppConfig.ppData.ROOT + 'data/fits/trace_' + current_stamp +
            '.npy', trace)
        return current_stamp, deposit, trace

    def __init__(self, rng, ppKernels, ppConfig):
        self.agents = np.zeros(
            shape=(ppConfig.ppData.N_AGENTS, 6),
            dtype=PPTypes.FLOAT_CPU)
        self.agents[:, 0] = rng.uniform(
            low=ppConfig.ppData.DOMAIN_MIN[0] + 0.001,
            high=ppConfig.ppData.DOMAIN_MAX[0] - 0.001,
            size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 1] = rng.uniform(
            low=ppConfig.ppData.DOMAIN_MIN[1] + 0.001,
            high=ppConfig.ppData.DOMAIN_MAX[1] - 0.001,
            size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 2] = rng.uniform(
            low=ppConfig.ppData.DOMAIN_MIN[2] + 0.001,
            high=ppConfig.ppData.DOMAIN_MAX[2] - 0.001,
            size=ppConfig.ppData.N_AGENTS)
        # zenith angle
        self.agents[:, 3] = np.arccos(2.0 * np.array(
            rng.uniform(low=0.0, high=1.0,
                        size=ppConfig.ppData.N_AGENTS)) - 1.0)
        # azimuth angle
        self.agents[:, 4] = rng.uniform(
            low=0.0, high=2.0 * np.pi,
            size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 5] = 1.0
        Logger.logToStdOut("info", 'Agent sample:', self.agents[0, :])

        self.data_field = ti.Vector.field(
            n=4,
            dtype=PPTypes.FLOAT_GPU,
            shape=ppConfig.ppData.N_DATA)
        self.agents_field = ti.Vector.field(
            n=6,
            dtype=PPTypes.FLOAT_GPU,
            shape=ppConfig.ppData.N_AGENTS)
        self.deposit_field = ti.Vector.field(
            n=2,
            dtype=PPTypes.FLOAT_GPU,
            shape=ppConfig.DEPOSIT_RESOLUTION)
        self.trace_field = ti.Vector.field(
            n=1,
            dtype=PPTypes.FLOAT_GPU,
            shape=ppConfig.TRACE_RESOLUTION)
        self.vis_field = ti.Vector.field(
            n=3,
            dtype=PPTypes.FLOAT_GPU,
            shape=ppConfig.VIS_RESOLUTION)
        Logger.logToStdOut(
            "info",
            'Total GPU memory allocated:', PPTypes.INT_CPU(
                4 * (
                    self.data_field.shape[0] * 4 +
                    self.agents_field.shape[0] * 6 +
                    self.deposit_field.shape[0] * self.deposit_field.shape[1] * 2 +
                    self.trace_field.shape[0] * self.trace_field.shape[1] * 1 +
                    self.vis_field.shape[0] * self.vis_field.shape[1] * 3
                ) / 2 ** 20), 'MB')
        self.ppConfig = ppConfig
        self.ppKernels = ppKernels
        self.__init_internal_data__(ppKernels)


class PPSimulation_3DDiscrete(PPSimulation):
    def __drawGUI__(self, window, ppConfig):
        GuiHelper.draw(self, window, ppConfig)

    def __init__(self, ppInternalData, ppConfig, batch_mode=False, num_iterations=-1):
        self.current_deposit_index = 0
        self.do_export = False
        self.do_screenshot = False
        self.do_quit = False
        self.do_simulate = True
        self.hide_UI = False

        camera_distance = 3.0 * ppConfig.DOMAIN_SIZE_MAX
        camera_polar = 0.5 * np.pi
        camera_azimuth = 0.0
        last_MMB = [-1.0, -1.0]
        last_RMB = [-1.0, -1.0]

        # Check if file exists
        if not os.path.exists("/tmp/flag"):
            if batch_mode is False:
                window = ti.ui.Window(
                    'PolyPhy',
                    (ppInternalData.vis_field.shape[0],
                     ppInternalData.vis_field.shape[1]),
                    show_window=True)
                window.show()
                canvas = window.get_canvas()

            curr_iteration = 0
            # Main simulation and rendering loop
            while window.running if 'window' in locals() else True:
                if batch_mode is True:
                    # Handle progress monitor
                    curr_iteration += 1
                    if curr_iteration > num_iterations:
                        break
                    if (num_iterations % curr_iteration) == 0:
                        Logger.logToStdOut(
                            "info",
                            'Running MCPM... iteration',
                            curr_iteration,
                            '/',
                            num_iterations)
                else:
                    # batch_mode is False
                    # Handle controls
                    if window.get_event(ti.ui.PRESS):
                        if window.event.key == 'e':
                            self.do_export = True
                        if window.event.key == 's':
                            self.do_screenshot = True
                        if window.event.key == 'h':
                            self.hide_UI = not self.hide_UI
                        if window.event.key in [ti.ui.ESCAPE]:
                            self.do_quit = True

                    # Handle camera control: rotation
                    mouse_pos = window.get_cursor_pos()
                    if window.is_pressed(ti.ui.RMB):
                        if last_RMB[0] < 0.0:
                            last_RMB = mouse_pos
                        else:
                            delta_RMB = np.subtract(mouse_pos, last_RMB)
                            last_RMB = mouse_pos
                            camera_azimuth -= 5.0 * delta_RMB[0]
                            camera_polar += 3.5 * delta_RMB[1]
                            camera_polar = np.min(
                                [np.max([1.0e-2, camera_polar]),
                                 np.pi-1.0e-2])
                    else:
                        last_RMB = [-1.0, -1.0]

                    # Handle camera control: zooming
                    if window.is_pressed(ti.ui.MMB):
                        if last_MMB[0] < 0.0:
                            last_MMB = mouse_pos
                        else:
                            delta_MMB = np.subtract(mouse_pos, last_MMB)
                            last_MMB = mouse_pos
                            camera_distance -= (
                                5.0 * ppConfig.DOMAIN_SIZE_MAX * delta_MMB[1])
                            camera_distance = np.max(
                                [camera_distance, 0.85 * ppConfig.DOMAIN_SIZE_MAX])
                    else:
                        last_MMB = [-1.0, -1.0]

                    if not self.hide_UI:
                        self.__drawGUI__(window, ppConfig)

                # Main simulation sequence
                if self.do_simulate:
                    ppInternalData.ppKernels.data_step_3D_discrete(
                        ppConfig.data_deposit,
                        self.current_deposit_index,
                        ppConfig.ppData.DOMAIN_MIN,
                        ppConfig.ppData.DOMAIN_MAX,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.data_field,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.agent_step_3D_discrete(
                        ppConfig.sense_distance,
                        ppConfig.sense_angle,
                        ppConfig.steering_rate,
                        ppConfig.sampling_exponent,
                        ppConfig.step_size,
                        ppConfig.agent_deposit,
                        self.current_deposit_index,
                        ppConfig.distance_sampling_distribution,
                        ppConfig.directional_sampling_distribution,
                        ppConfig.directional_mutation_type,
                        ppConfig.deposit_fetching_strategy,
                        ppConfig.agent_boundary_handling,
                        ppConfig.ppData.N_DATA,
                        ppConfig.ppData.N_AGENTS,
                        ppConfig.ppData.DOMAIN_SIZE,
                        ppConfig.ppData.DOMAIN_MIN,
                        ppConfig.ppData.DOMAIN_MAX,
                        ppConfig.TRACE_RESOLUTION,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.agents_field,
                        ppInternalData.trace_field,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.deposit_relaxation_step_3D_discrete(
                        ppConfig.deposit_attenuation,
                        self.current_deposit_index,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.trace_relaxation_step_3D_discrete(
                        ppConfig.trace_attenuation, ppInternalData.trace_field)
                    self.current_deposit_index = 1 - self.current_deposit_index

                # Render visualization
                ppInternalData.ppKernels.render_visualization_3D_raymarched(
                    ppConfig.trace_vis,
                    ppConfig.deposit_vis,
                    camera_distance,
                    camera_polar,
                    camera_azimuth,
                    ppConfig.n_ray_steps,
                    self.current_deposit_index,
                    ppConfig.TRACE_RESOLUTION,
                    ppConfig.DEPOSIT_RESOLUTION,
                    ppConfig.VIS_RESOLUTION,
                    ppConfig.DOMAIN_SIZE_MAX,
                    ppConfig.ppData.DOMAIN_MIN,
                    ppConfig.ppData.DOMAIN_MAX,
                    ppConfig.ppData.DOMAIN_CENTER,
                    ppConfig.RAY_EPSILON,
                    ppInternalData.deposit_field,
                    ppInternalData.trace_field,
                    ppInternalData.vis_field)

                if batch_mode is False:
                    canvas.set_image(ppInternalData.vis_field)
                    if self.do_screenshot:
                        window.save_image(
                            ppConfig.ppData.ROOT + 'capture/screenshot_'
                            + Logger.stamp() + '.png')
                        # Must appear before window.show() call
                        self.do_screenshot = False
                    window.show()
                if self.do_export:
                    ppInternalData.store_fit()
                    self.do_export = False
                if self.do_quit:
                    break

            if not batch_mode:
                window.destroy()


class PPPostSimulation_3DDiscrete(PPPostSimulation):
    def __init__(self, ppInternalData):
        super().__init__(ppInternalData)
        ppInternalData.store_fit()
