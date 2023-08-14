import numpy as np
import math, os
from numpy.random import default_rng
import time
from datetime import datetime
import matplotlib.pyplot as plt
import taichi as ti
import taichi.math as timath

from first import TypeAliases
from second import PolyphyEnums
from third import SimulationConstants, StateFlags

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
                    window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.derivedVariables.VIS_RESOLUTION[0]), 0.74 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.derivedVariables.VIS_RESOLUTION[1]))
                    window.GUI.text("MCPM parameters:")
                    simulationVisuals.sense_distance = window.GUI.slider_float('Sensing dist', simulationVisuals.sense_distance, 0.1, 0.05 * np.max([simulationVisuals.dataLoaders.DOMAIN_SIZE[0], simulationVisuals.dataLoaders.DOMAIN_SIZE[1]]))
                    simulationVisuals.sense_angle = window.GUI.slider_float('Sensing angle', simulationVisuals.sense_angle, 0.01, 0.5 * np.pi)
                    simulationVisuals.sampling_exponent = window.GUI.slider_float('Sampling expo', simulationVisuals.sampling_exponent, 1.0, 10.0)
                    simulationVisuals.step_size = window.GUI.slider_float('Step size', simulationVisuals.step_size, 0.0, 0.005 * np.max([simulationVisuals.dataLoaders.DOMAIN_SIZE[0], simulationVisuals.dataLoaders.DOMAIN_SIZE[1]]))
                    simulationVisuals.data_deposit = window.GUI.slider_float('Data deposit', simulationVisuals.data_deposit, 0.0, SimulationConstants.MAX_DEPOSIT)
                    simulationVisuals.agent_deposit = window.GUI.slider_float('Agent deposit', simulationVisuals.agent_deposit, 0.0, 10.0 * SimulationConstants.MAX_DEPOSIT * simulationVisuals.derivedVariables.DATA_TO_AGENTS_RATIO)
                    simulationVisuals.deposit_attenuation = window.GUI.slider_float('Deposit attn', simulationVisuals.deposit_attenuation, 0.8, 0.999)
                    simulationVisuals.trace_attenuation = window.GUI.slider_float('Trace attn', simulationVisuals.trace_attenuation, 0.8, 0.999)
                    simulationVisuals.deposit_vis = math.pow(10.0, window.GUI.slider_float('Deposit vis', math.log(simulationVisuals.deposit_vis, 10.0), -3.0, 3.0))
                    simulationVisuals.trace_vis = math.pow(10.0, window.GUI.slider_float('Trace vis', math.log(simulationVisuals.trace_vis, 10.0), -3.0, 3.0))
            
                    window.GUI.text("Distance distribution:")
                    if window.GUI.checkbox("Constant", StateFlags.distance_sampling_distribution == PolyphyEnums.EnumDistanceSamplingDistribution.CONSTANT):
                        StateFlags.distance_sampling_distribution = PolyphyEnums.EnumDistanceSamplingDistribution.CONSTANT
                    if window.GUI.checkbox("Exponential", StateFlags.distance_sampling_distribution == PolyphyEnums.EnumDistanceSamplingDistribution.EXPONENTIAL):
                        StateFlags.distance_sampling_distribution = PolyphyEnums.EnumDistanceSamplingDistribution.EXPONENTIAL
                    if window.GUI.checkbox("Maxwell-Boltzmann", StateFlags.distance_sampling_distribution == PolyphyEnums.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
                        StateFlags.distance_sampling_distribution = PolyphyEnums.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
            
                    window.GUI.text("Directional distribution:")
                    if window.GUI.checkbox("Discrete", StateFlags.directional_sampling_distribution == PolyphyEnums.EnumDirectionalSamplingDistribution.DISCRETE):
                        StateFlags.directional_sampling_distribution = PolyphyEnums.EnumDirectionalSamplingDistribution.DISCRETE
                    if window.GUI.checkbox("Cone", StateFlags.directional_sampling_distribution == PolyphyEnums.EnumDirectionalSamplingDistribution.CONE):
                        StateFlags.directional_sampling_distribution = PolyphyEnums.EnumDirectionalSamplingDistribution.CONE
            
                    window.GUI.text("Directional mutation:")
                    if window.GUI.checkbox("Deterministic", StateFlags.directional_mutation_type == PolyphyEnums.EnumDirectionalMutationType.DETERMINISTIC):
                        StateFlags.directional_mutation_type = PolyphyEnums.EnumDirectionalMutationType.DETERMINISTIC
                    if window.GUI.checkbox("Stochastic", StateFlags.directional_mutation_type == PolyphyEnums.EnumDirectionalMutationType.PROBABILISTIC):
                        StateFlags.directional_mutation_type = PolyphyEnums.EnumDirectionalMutationType.PROBABILISTIC
            
                    window.GUI.text("Deposit fetching:")
                    if window.GUI.checkbox("Nearest neighbor", StateFlags.deposit_fetching_strategy == PolyphyEnums.EnumDepositFetchingStrategy.NN):
                        StateFlags.deposit_fetching_strategy = PolyphyEnums.EnumDepositFetchingStrategy.NN
                    if window.GUI.checkbox("Noise-perturbed NN", StateFlags.deposit_fetching_strategy == PolyphyEnums.EnumDepositFetchingStrategy.NN_PERTURBED):
                        StateFlags.deposit_fetching_strategy = PolyphyEnums.EnumDepositFetchingStrategy.NN_PERTURBED
            
                    window.GUI.text("Agent boundary handling:")
                    if window.GUI.checkbox("Wrap around", StateFlags.agent_boundary_handling == PolyphyEnums.EnumAgentBoundaryHandling.WRAP):
                        StateFlags.agent_boundary_handling = PolyphyEnums.EnumAgentBoundaryHandling.WRAP
                    if window.GUI.checkbox("Reinitialize center", StateFlags.agent_boundary_handling == PolyphyEnums.EnumAgentBoundaryHandling.REINIT_CENTER):
                        StateFlags.agent_boundary_handling = PolyphyEnums.EnumAgentBoundaryHandling.REINIT_CENTER
                    if window.GUI.checkbox("Reinitialize randomly", StateFlags.agent_boundary_handling == PolyphyEnums.EnumAgentBoundaryHandling.REINIT_RANDOMLY):
                        StateFlags.agent_boundary_handling = PolyphyEnums.EnumAgentBoundaryHandling.REINIT_RANDOMLY
            
                    window.GUI.text("Misc controls:")
                    simulationVisuals.do_simulate = window.GUI.checkbox("Run simulation", simulationVisuals.do_simulate)
                    do_export = do_export | window.GUI.button('Export fit')
                    do_screenshot = do_screenshot | window.GUI.button('Screenshot')
                    do_quit = do_quit | window.GUI.button('Quit')
                    window.GUI.end()
            
                    ## Help window
                    ## Do not exceed prescribed line length of 120 characters, there is no text wrapping in Taichi GUI for now
                    window.GUI.begin('Help', 0.35 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.derivedVariables.VIS_RESOLUTION[0]), 0.01, 0.6, 0.30 * 1024.0 / TypeAliases.FLOAT_CPU(simulationVisuals.derivedVariables.VIS_RESOLUTION[1]))
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
                    k.data_step(simulationVisuals.data_deposit, simulationVisuals.current_deposit_index)
                    k.agent_step(\
                        simulationVisuals.sense_distance,\
                        simulationVisuals.sense_angle,\
                        SimulationConstants.STEERING_RATE,\
                        simulationVisuals.sampling_exponent,\
                        simulationVisuals.step_size,\
                        simulationVisuals.agent_deposit,\
                        simulationVisuals.current_deposit_index,\
                        StateFlags.distance_sampling_distribution,\
                        StateFlags.directional_sampling_distribution,\
                        StateFlags.directional_mutation_type,\
                        StateFlags.deposit_fetching_strategy,\
                        StateFlags.agent_boundary_handling)
                    k.deposit_relaxation_step(simulationVisuals.deposit_attenuation, simulationVisuals.current_deposit_index)
                    k.trace_relaxation_step(simulationVisuals.trace_attenuation)
                    simulationVisuals.current_deposit_index = 1 - simulationVisuals.current_deposit_index
            
                ## Render visualization
                k.render_visualization(simulationVisuals.deposit_vis, simulationVisuals.trace_vis, simulationVisuals.current_deposit_index)
                canvas.set_image(simulationVisuals.fieldVariables.vis_field)
            
                if do_screenshot:
                    window.write_image(simulationVisuals.dataLoaders.ROOT + 'capture/screenshot_' + simulationVisuals.stamp() + '.png') ## Must appear before window.show() call
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
        np.save(simulationVisuals.dataLoaders.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = simulationVisuals.fieldVariables.trace_field.to_numpy()
        np.save(simulationVisuals.dataLoaders.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)

        ## Plot results
        ## Compare with stored fields
        current_stamp, deposit, trace = simulationVisuals.store_fit()

        plt.figure(figsize = (10.0, 10.0))
        plt.imshow(np.flip(np.transpose(deposit[:,:,0]), axis=0))
        plt.figure(figsize = (10.0, 10.0))
        deposit_restored = np.load(simulationVisuals.dataLoaders.ROOT + 'data/fits/deposit_' + current_stamp + '.npy')
        plt.imshow(np.flip(np.transpose(deposit_restored[:,:,0]), axis=0))

        plt.figure(figsize = (10.0, 10.0))
        plt.imshow(np.flip(np.transpose(trace[:,:,0]), axis=0))
        plt.figure(figsize = (10.0, 10.0))
        trace_restored = np.load(simulationVisuals.dataLoaders.ROOT + 'data/fits/trace_' + current_stamp + '.npy')
        plt.imshow(np.flip(np.transpose(trace_restored[:,:,0]), axis=0))

