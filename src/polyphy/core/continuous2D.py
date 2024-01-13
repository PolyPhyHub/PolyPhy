# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import os
import numpy as np
import taichi as ti

from .common import PPTypes, PPConfig, PPInputData, PPInternalData
from .common import PPSimulation, PPPostSimulation
from utils.gui_helper import GuiHelper
from utils.logger import Logger

from PIL import Image


class PPConfig_2DContinuous(PPConfig):
    def __init__(self):
        super()

    def register_data(self, ppData):
        self.ppData = ppData
        self.DATA_TO_AGENTS_RATIO = 1.0e4 / PPTypes.FLOAT_CPU(ppData.N_AGENTS)
        self.DOMAIN_SIZE_MAX = np.max([ppData.DOMAIN_SIZE[0], ppData.DOMAIN_SIZE[1]])
        self.TRACE_RESOLUTION = PPTypes.INT_CPU(
            (PPTypes.FLOAT_CPU(self.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[0] /
                self.DOMAIN_SIZE_MAX, PPTypes.FLOAT_CPU(
                self.TRACE_RESOLUTION_MAX) * ppData.DOMAIN_SIZE[1] /
                self.DOMAIN_SIZE_MAX)
        )
        self.DEPOSIT_RESOLUTION = (
            self.TRACE_RESOLUTION[0] //
            PPConfig.DEPOSIT_DOWNSCALING_FACTOR, self.TRACE_RESOLUTION[1] //
            PPConfig.DEPOSIT_DOWNSCALING_FACTOR
        )

        # Check if these are set and if not give them decent initial estimates
        if self.sense_distance < 1.e-4:
            self.sense_distance = 0.005 * self.DOMAIN_SIZE_MAX
        if self.step_size < 1.e-5:
            self.step_size = 0.0005 * self.DOMAIN_SIZE_MAX
        if self.data_deposit < -1.e-4:
            self.data_deposit = 0.1 * PPConfig.MAX_DEPOSIT
        if self.agent_deposit < -1.e-5:
            self.agent_deposit = self.data_deposit * self.DATA_TO_AGENTS_RATIO
        self.VIS_RESOLUTION = self.TRACE_RESOLUTION
        self.input_file = ppData.input_file
        Logger.logToStdOut('info', 'Trace grid resolution:', self.TRACE_RESOLUTION)
        Logger.logToStdOut('info', 'Deposit grid resolution:', self.DEPOSIT_RESOLUTION)


class PPInputData_2DContinuous(PPInputData):
    # TODO: load datasets from specified file + type

    def __load_from_file__(self):
        Logger.logToStdOut("info", 'Loading input file... ' + self.ROOT + self.input_file)
        self.data = ti.tools.image.imread(self.ROOT + self.input_file).astype(PPTypes.FLOAT_CPU) / 255.0
        self.DATA_RESOLUTION = self.data.shape[0:2]
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.DOMAIN_SIZE = (PPConfig.DOMAIN_SIZE_DEFAULT, PPConfig.DOMAIN_SIZE_DEFAULT * PPTypes.FLOAT_CPU(self.DATA_RESOLUTION[1]) / PPTypes.FLOAT_CPU(self.DATA_RESOLUTION[0]))
        self.DOMAIN_MIN = (0.0, 0.0)
        self.DOMAIN_MAX = self.DOMAIN_SIZE
        self.AVG_WEIGHT = 1.0

    def __generate_test_data__(self, rng):
        Logger.logToStdOut("info", 'Generating synthetic testing dataset...')
        self.data = np.ones(shape=(PPConfig.TRACE_RESOLUTION_MAX, PPConfig.TRACE_RESOLUTION_MAX, 1), dtype=PPTypes.FLOAT_CPU)
        self.data = rng.uniform(low=0.0, high=2.0, size=self.data.shape).astype(dtype=PPTypes.FLOAT_CPU)
        self.DATA_RESOLUTION = (PPConfig.TRACE_RESOLUTION_MAX, PPConfig.TRACE_RESOLUTION_MAX)
        self.N_AGENTS = PPConfig.N_AGENTS_DEFAULT
        self.DOMAIN_SIZE = (PPConfig.DOMAIN_SIZE_DEFAULT, PPConfig.DOMAIN_SIZE_DEFAULT
                            * PPTypes.FLOAT_CPU(self.DATA_RESOLUTION[1]) / PPTypes.FLOAT_CPU(self.DATA_RESOLUTION[0]))
        self.DOMAIN_MIN = (0.0, 0.0)
        self.DOMAIN_MAX = self.DOMAIN_SIZE
        self.AVG_WEIGHT = 1.0


class PPInternalData_2DContinuous(PPInternalData):
    def __init_internal_data__(self, ppKernels):
        # Initialize GPU fields
        self.data_field.from_numpy(self.ppConfig.ppData.data)
        self.agents_field.from_numpy(self.agents)
        ppKernels.zero_field(self.deposit_field)
        ppKernels.zero_field(self.trace_field)
        ppKernels.zero_field(self.vis_field)

    # TODO implement meaningful editing using 2D discrete pipeline as template
    def edit_data(self, edit_index: PPTypes.INT_CPU, window: ti.ui.Window) -> PPTypes.INT_CPU:
        return 0

    def __init__(self, rng, ppKernels, ppConfig):
        self.agents = np.zeros(
            shape=(ppConfig.ppData.N_AGENTS, 4), dtype=PPTypes.FLOAT_CPU)
        self.agents[:, 0] = rng.uniform(low=ppConfig.ppData.DOMAIN_MIN[0] + 0.001,
                                        high=ppConfig.ppData.DOMAIN_MAX[0] - 0.001,
                                        size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 1] = rng.uniform(low=ppConfig.ppData.DOMAIN_MIN[1] + 0.001,
                                        high=ppConfig.ppData.DOMAIN_MAX[1] - 0.001,
                                        size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 2] = rng.uniform(low=0.0, high=2.0 * np.pi,
                                        size=ppConfig.ppData.N_AGENTS)
        self.agents[:, 3] = 1.0
        Logger.logToStdOut("info", 'Agent sample:', self.agents[0, :])

        self.data_field = ti.Vector.field(n=1, dtype=PPTypes.FLOAT_GPU,
                                          shape=ppConfig.ppData.DATA_RESOLUTION)
        self.agents_field = ti.Vector.field(n=4, dtype=PPTypes.FLOAT_GPU,
                                            shape=ppConfig.ppData.N_AGENTS)
        self.deposit_field = ti.Vector.field(n=2, dtype=PPTypes.FLOAT_GPU,
                                             shape=ppConfig.DEPOSIT_RESOLUTION)
        self.trace_field = ti.Vector.field(n=1, dtype=PPTypes.FLOAT_GPU,
                                           shape=ppConfig.TRACE_RESOLUTION)
        self.vis_field = ti.Vector.field(n=3, dtype=PPTypes.FLOAT_GPU,
                                         shape=ppConfig.VIS_RESOLUTION)
        Logger.logToStdOut("info", 'Total GPU memory allocated:',
                           PPTypes.INT_CPU(4 * (self.data_field.shape[0] * self.data_field.shape[1] +
                                                self.agents_field.shape[0] * 4 +
                                                self.deposit_field.shape[0] *
                                                self.deposit_field.shape[1] * 2 +
                                                self.trace_field.shape[0] *
                                                self.trace_field.shape[1] * 1 +
                                                self.vis_field.shape[0] *
                                                self.vis_field.shape[1] * 3
                                                ) / 2 ** 20), 'MB')

        self.ppConfig = ppConfig
        self.ppKernels = ppKernels
        self.__init_internal_data__(ppKernels)


class PPSimulation_2DContinuous(PPSimulation):
    def __drawGUI__(self, window, ppConfig):
        GuiHelper.draw(self, window, ppConfig)

    def __init__(self, ppInternalData, ppConfig, batch_mode=False, num_iterations=-1):
        self.current_deposit_index = 0
        self.data_edit_index = 0

        self.do_export = False
        self.do_screenshot = False
        self.do_quit = False
        self.do_simulate = True
        self.hide_UI = False

        # Check if file exists
        if not os.path.exists("/tmp/flag"):
            if batch_mode is False:
                window = ti.ui.Window('PolyPhy', (ppInternalData.vis_field.shape[0],
                                                  ppInternalData.vis_field.shape[1]
                                                  ), show_window=True)
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
                        Logger.logToStdOut("info", 'Running MCPM... iteration',
                                           curr_iteration, '/', num_iterations)
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
                        if window.event.key in [ti.ui.LMB]:
                            self.data_edit_index = ppInternalData.edit_data(
                                self.data_edit_index, window)
                    if window.is_pressed(ti.ui.RMB):
                        self.data_edit_index = ppInternalData.edit_data(
                            self.data_edit_index, window)
                    if not self.hide_UI:
                        self.__drawGUI__(window, ppConfig)

                # Main simulation sequence
                if self.do_simulate:
                    ppInternalData.ppKernels.data_step_2D_continuous(
                        ppConfig.data_deposit,
                        self.current_deposit_index,
                        ppConfig.ppData.DOMAIN_MIN,
                        ppConfig.ppData.DOMAIN_MAX,
                        ppConfig.ppData.DOMAIN_SIZE,
                        ppConfig.ppData.DATA_RESOLUTION,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.data_field,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.agent_step_2D(
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
                        10000,
                        ppConfig.ppData.N_AGENTS,
                        ppConfig.ppData.DOMAIN_SIZE,
                        ppConfig.ppData.DOMAIN_MIN,
                        ppConfig.ppData.DOMAIN_MAX,
                        ppConfig.TRACE_RESOLUTION,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.agents_field,
                        ppInternalData.trace_field,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.deposit_relaxation_step_2D(
                        ppConfig.deposit_attenuation,
                        self.current_deposit_index,
                        ppConfig.DEPOSIT_RESOLUTION,
                        ppInternalData.deposit_field)
                    ppInternalData.ppKernels.trace_relaxation_step_2D(
                        ppConfig.trace_attenuation,
                        ppInternalData.trace_field)
                    self.current_deposit_index = 1 - self.current_deposit_index

                # Render visualization
                ppInternalData.ppKernels.render_visualization_2D(
                    ppConfig.trace_vis,
                    ppConfig.deposit_vis,
                    self.current_deposit_index,
                    ppConfig.TRACE_RESOLUTION,
                    ppConfig.DEPOSIT_RESOLUTION,
                    ppConfig.VIS_RESOLUTION,
                    ppInternalData.trace_field,
                    ppInternalData.deposit_field,
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

            if batch_mode is False:
                window.destroy()


class PPPostSimulation_2DContinuous(PPPostSimulation):
    def __init__(self, ppInternalData):
        super().__init__(ppInternalData)
        ppInternalData.store_fit()
