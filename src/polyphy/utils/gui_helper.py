import numpy as np
import math

from core.common import PPTypes


class GuiHelper:
    @staticmethod
    def draw(self, window, ppConfig):
        # Draw main interactive control GUI
        window.GUI.begin(
            'Main', 0.01, 0.01, 0.32 * 1024.0 /
            PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[0]), 0.74 * 1024.0 /
            PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[1]))
        window.GUI.text("MCPM parameters:")
        ppConfig.sense_distance = window.GUI.slider_float(
            'Sensing dist', ppConfig.sense_distance,
            0.1, 0.05 * ppConfig.DOMAIN_SIZE_MAX)
        ppConfig.sense_angle = window.GUI.slider_float(
            'Sensing angle', ppConfig.sense_angle, 0.01, 0.5 * np.pi)
        ppConfig.sampling_exponent = window.GUI.slider_float(
            'Sampling expo', ppConfig.sampling_exponent, 0.1, 5.0)
        ppConfig.step_size = window.GUI.slider_float(
            'Step size', ppConfig.step_size,
            0.0, 0.005 * ppConfig.DOMAIN_SIZE_MAX)
        ppConfig.data_deposit = window.GUI.slider_float(
            'Data deposit', ppConfig.data_deposit, 0.0, ppConfig.MAX_DEPOSIT)
        ppConfig.agent_deposit = window.GUI.slider_float(
            'Agent deposit', ppConfig.agent_deposit, 0.0,
            10.0 * ppConfig.MAX_DEPOSIT * ppConfig.DATA_TO_AGENTS_RATIO)
        ppConfig.deposit_attenuation = window.GUI.slider_float(
            'Deposit attn', ppConfig.deposit_attenuation, 0.8, 0.999)
        ppConfig.trace_attenuation = window.GUI.slider_float(
            'Trace attn', ppConfig.trace_attenuation, 0.8, 0.999)
        ppConfig.deposit_vis = math.pow(10.0, window.GUI.slider_float(
            'Deposit vis', math.log(ppConfig.deposit_vis, 10.0), -3.0, 3.0))
        ppConfig.trace_vis = math.pow(10.0, window.GUI.slider_float(
            'Trace vis', math.log(ppConfig.trace_vis, 10.0), -3.0, 3.0))

        window.GUI.text("Distance distribution:")
        if window.GUI.checkbox(
            "Constant", ppConfig.distance_sampling_distribution
                == ppConfig.EnumDistanceSamplingDistribution.CONSTANT):
            ppConfig.distance_sampling_distribution = \
                ppConfig.EnumDistanceSamplingDistribution.CONSTANT
        if window.GUI.checkbox(
            "Exponential", ppConfig.distance_sampling_distribution
                == ppConfig.EnumDistanceSamplingDistribution.EXPONENTIAL):
            ppConfig.distance_sampling_distribution = \
                ppConfig.EnumDistanceSamplingDistribution.EXPONENTIAL
        if window.GUI.checkbox(
            "Maxwell-Boltzmann", ppConfig.distance_sampling_distribution
                == ppConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
            ppConfig.distance_sampling_distribution = \
                ppConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
        window.GUI.text("Directional distribution:")
        if window.GUI.checkbox(
            "Discrete", ppConfig.directional_sampling_distribution
                == ppConfig.EnumDirectionalSamplingDistribution.DISCRETE):
            ppConfig.directional_sampling_distribution = \
                ppConfig.EnumDirectionalSamplingDistribution.DISCRETE
        if window.GUI.checkbox(
            "Cone", ppConfig.directional_sampling_distribution
                == ppConfig.EnumDirectionalSamplingDistribution.CONE):
            ppConfig.directional_sampling_distribution = \
                ppConfig.EnumDirectionalSamplingDistribution.CONE
        window.GUI.text("Directional mutation:")
        if window.GUI.checkbox(
            "Deterministic", ppConfig.directional_mutation_type
                == ppConfig.EnumDirectionalMutationType.DETERMINISTIC):
            ppConfig.directional_mutation_type = \
                ppConfig.EnumDirectionalMutationType.DETERMINISTIC
        if window.GUI.checkbox(
            "Stochastic", ppConfig.directional_mutation_type
                == ppConfig.EnumDirectionalMutationType.PROBABILISTIC):
            ppConfig.directional_mutation_type = \
                ppConfig.EnumDirectionalMutationType.PROBABILISTIC
        window.GUI.text("Deposit fetching:")
        if window.GUI.checkbox(
            "Nearest neighbor", ppConfig.deposit_fetching_strategy
                == ppConfig.EnumDepositFetchingStrategy.NN):
            ppConfig.deposit_fetching_strategy = \
                ppConfig.EnumDepositFetchingStrategy.NN
        if window.GUI.checkbox(
            "Noise-perturbed NN", ppConfig.deposit_fetching_strategy
                == ppConfig.EnumDepositFetchingStrategy.NN_PERTURBED):
            ppConfig.deposit_fetching_strategy = \
                ppConfig.EnumDepositFetchingStrategy.NN_PERTURBED
        window.GUI.text("Agent boundary handling:")
        if window.GUI.checkbox(
            "Wrap around", ppConfig.agent_boundary_handling
                == ppConfig.EnumAgentBoundaryHandling.WRAP):
            ppConfig.agent_boundary_handling = \
                ppConfig.EnumAgentBoundaryHandling.WRAP
        if window.GUI.checkbox(
            "Reinitialize center", ppConfig.agent_boundary_handling
                == ppConfig.EnumAgentBoundaryHandling.REINIT_CENTER):
            ppConfig.agent_boundary_handling = \
                ppConfig.EnumAgentBoundaryHandling.REINIT_CENTER
        if window.GUI.checkbox(
            "Reinitialize randomly", ppConfig.agent_boundary_handling
                == ppConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY):
            ppConfig.agent_boundary_handling = \
                ppConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY

        window.GUI.text("Misc controls:")
        self.do_simulate = window.GUI.checkbox("Run simulation", self.do_simulate)
        self.do_export = self.do_export | window.GUI.button('Export fit')
        self.do_screenshot = self.do_screenshot | window.GUI.button('Screenshot')
        self.do_quit = self.do_quit | window.GUI.button('Quit')
        window.GUI.end()

        # Help window
        # Do not exceed prescribed line length of 120 characters,
        # there is no text wrapping in Taichi GUI
        window.GUI.begin('Help', 0.35 * 1024.0 / PPTypes.FLOAT_CPU(
            ppConfig.VIS_RESOLUTION[0]),
            0.01, 0.6, 0.30 * 1024.0 / PPTypes.FLOAT_CPU(ppConfig.VIS_RESOLUTION[1]))
        window.GUI.text("Welcome to PolyPhy 2D GUI variant written by researchers at \
                        UCSC/OSPO with the help of numerous external contributors\n\
                        (https://github.com/PolyPhyHub).\
                        PolyPhy implements MCPM, an agent-based, stochastic,\
                        pattern forming algorithm designed\nby Elek et al, inspired by\
                        Physarum polycephalum slime mold. Below is a quick reference\
                        guide explaining the parameters\nand features available\
                         in the interface. The reference as well as other panels can be\
                         hidden using the arrow button, moved,\nand rescaled.")
        window.GUI.text("")
        window.GUI.text("PARAMETERS")
        window.GUI.text("Sensing dist: average distance in world units at which agents\
                         probe the deposit")
        window.GUI.text("Sensing angle: angle in radians within which agents probe\
                         deposit (left and right concentric to movement direction)")
        window.GUI.text("Sampling expo: sampling sharpness \
                        (or 'acuteness' or 'temperature') \
                        which tunes the directional mutation behavior")
        window.GUI.text("Step size: average size of the step in world units which\
                         agents make in each iteration")
        window.GUI.text("Data deposit: amount of marker 'deposit' that *data* emit at\
                         every iteration")
        window.GUI.text("Agent deposit: amount of marker 'deposit' that *agents* emit\
                         at every iteration")
        window.GUI.text("Deposit attn: attenuation (or 'decay') rate of the diffusing\
                         combined agent+data deposit field")
        window.GUI.text("Trace attn: attenuation (or 'decay') of the non-diffusing\
                         agent trace field")
        window.GUI.text("Deposit vis: visualization intensity of the green deposit\
                         field (logarithmic)")
        window.GUI.text("Trace vis: visualization intensity of the red trace field\
                         (logarithmic)")
        window.GUI.text("")
        window.GUI.text("OPTIONS")
        window.GUI.text("Distance distribution: strategy for sampling the sensing and\
                         movement distances")
        window.GUI.text("Directional distribution: strategy for sampling the sensing\
                         and movement directions")
        window.GUI.text("Directional mutation: strategy for selecting the new movement\
                         direction")
        window.GUI.text("Deposit fetching: access behavior when sampling the deposit\
                         field")
        window.GUI.text("Agent boundary handling: what do agents do if they reach the\
                         boundary of the simulation domain")
        window.GUI.text("")
        window.GUI.text("VISUALIZATION")
        window.GUI.text("Renders 2 types of information superimposed on top of each\
                         other: *green* deposit field and *red-purple* trace field.")
        window.GUI.text("Yellow-white signifies areas where deposit and trace overlap\
                         (relative intensities are controlled by the T/D vis params)")
        window.GUI.text("Screenshots can be saved in the /capture folder.")
        window.GUI.text("")
        window.GUI.text("DATA")
        window.GUI.text("Input data are loaded from the specified folder in /data.\
                         Currently the CSV format is supported.")
        window.GUI.text("Reconstruction data are exported to /data/fits using the\
                         Export fit button.")
        window.GUI.text("")
        window.GUI.text("EDITING")
        window.GUI.text("New data points can be placed by mouse clicking.\
                         This overrides old data on a Round-Robin basis.")
        window.GUI.text("Left mouse: discrete mode, place a single data point")
        window.GUI.text("Right mouse: continuous mode,\
                         place a data point at every iteration")
        window.GUI.end()
