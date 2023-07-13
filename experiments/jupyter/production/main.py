import numpy as np
import math, os
from numpy.random import default_rng
import time
from datetime import datetime
import matplotlib.pyplot as plt
import taichi as ti
import taichi.math as timath

import first
import second
import third
import fourth
import final

## check if file exists
if os.path.exists("/tmp/flag") == False:
    window = ti.ui.Window('PolyPhy', (fourth.vis_field.shape[0], fourth.vis_field.shape[1]), show_window = True)
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
                final.data_edit_index = final.edit_data(final.data_edit_index)
        if window.is_pressed(ti.ui.RMB):
            final.data_edit_index = final.edit_data(final.data_edit_index)
        
        if not final.hide_UI:
            ## Draw main interactive control GUI
            window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[0]), 0.74 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[1]))
            window.GUI.text("MCPM parameters:")
            final.sense_distance = window.GUI.slider_float('Sensing dist', final.sense_distance, 0.1, 0.05 * np.max([fourth.DOMAIN_SIZE[0], fourth.DOMAIN_SIZE[1]]))
            final.sense_angle = window.GUI.slider_float('Sensing angle', final.sense_angle, 0.01, 0.5 * np.pi)
            final.sampling_exponent = window.GUI.slider_float('Sampling expo', final.sampling_exponent, 1.0, 10.0)
            final.step_size = window.GUI.slider_float('Step size', final.step_size, 0.0, 0.005 * np.max([fourth.DOMAIN_SIZE[0], fourth.DOMAIN_SIZE[1]]))
            final.data_deposit = window.GUI.slider_float('Data deposit', final.data_deposit, 0.0, third.MAX_DEPOSIT)
            final.agent_deposit = window.GUI.slider_float('Agent deposit', final.agent_deposit, 0.0, 10.0 * third.MAX_DEPOSIT * fourth.DATA_TO_AGENTS_RATIO)
            final.deposit_attenuation = window.GUI.slider_float('Deposit attn', final.deposit_attenuation, 0.8, 0.999)
            final.trace_attenuation = window.GUI.slider_float('Trace attn', final.trace_attenuation, 0.8, 0.999)
            final.deposit_vis = math.pow(10.0, window.GUI.slider_float('Deposit vis', math.log(final.deposit_vis, 10.0), -3.0, 3.0))
            final.trace_vis = math.pow(10.0, window.GUI.slider_float('Trace vis', math.log(final.trace_vis, 10.0), -3.0, 3.0))
    
            window.GUI.text("Distance distribution:")
            if window.GUI.checkbox("Constant", third.distance_sampling_distribution == second.EnumDistanceSamplingDistribution.CONSTANT):
                third.distance_sampling_distribution = second.EnumDistanceSamplingDistribution.CONSTANT
            if window.GUI.checkbox("Exponential", third.distance_sampling_distribution == second.EnumDistanceSamplingDistribution.EXPONENTIAL):
                third.distance_sampling_distribution = second.EnumDistanceSamplingDistribution.EXPONENTIAL
            if window.GUI.checkbox("Maxwell-Boltzmann", third.distance_sampling_distribution == second.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
                third.distance_sampling_distribution = second.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
    
            window.GUI.text("Directional distribution:")
            if window.GUI.checkbox("Discrete", third.directional_sampling_distribution == second.EnumDirectionalSamplingDistribution.DISCRETE):
                third.directional_sampling_distribution = second.EnumDirectionalSamplingDistribution.DISCRETE
            if window.GUI.checkbox("Cone", third.directional_sampling_distribution == second.EnumDirectionalSamplingDistribution.CONE):
                third.directional_sampling_distribution = second.EnumDirectionalSamplingDistribution.CONE
    
            window.GUI.text("Directional mutation:")
            if window.GUI.checkbox("Deterministic", third.directional_mutation_type == second.EnumDirectionalMutationType.DETERMINISTIC):
                third.directional_mutation_type = second.EnumDirectionalMutationType.DETERMINISTIC
            if window.GUI.checkbox("Stochastic", third.directional_mutation_type == second.EnumDirectionalMutationType.PROBABILISTIC):
                third.directional_mutation_type = second.EnumDirectionalMutationType.PROBABILISTIC
    
            window.GUI.text("Deposit fetching:")
            if window.GUI.checkbox("Nearest neighbor", third.deposit_fetching_strategy == second.EnumDepositFetchingStrategy.NN):
                third.deposit_fetching_strategy = second.EnumDepositFetchingStrategy.NN
            if window.GUI.checkbox("Noise-perturbed NN", third.deposit_fetching_strategy == second.EnumDepositFetchingStrategy.NN_PERTURBED):
                third.deposit_fetching_strategy = second.EnumDepositFetchingStrategy.NN_PERTURBED
    
            window.GUI.text("Agent boundary handling:")
            if window.GUI.checkbox("Wrap around", third.agent_boundary_handling == second.EnumAgentBoundaryHandling.WRAP):
                third.agent_boundary_handling = second.EnumAgentBoundaryHandling.WRAP
            if window.GUI.checkbox("Reinitialize center", third.agent_boundary_handling == second.EnumAgentBoundaryHandling.REINIT_CENTER):
                third.agent_boundary_handling = second.EnumAgentBoundaryHandling.REINIT_CENTER
            if window.GUI.checkbox("Reinitialize randomly", third.agent_boundary_handling == second.EnumAgentBoundaryHandling.REINIT_RANDOMLY):
                third.agent_boundary_handling = second.EnumAgentBoundaryHandling.REINIT_RANDOMLY
    
            window.GUI.text("Misc controls:")
            final.do_simulate = window.GUI.checkbox("Run simulation", final.do_simulate)
            do_export = do_export | window.GUI.button('Export fit')
            do_screenshot = do_screenshot | window.GUI.button('Screenshot')
            do_quit = do_quit | window.GUI.button('Quit')
            window.GUI.end()
    
            ## Help window
            ## Do not exceed prescribed line length of 120 characters, there is no text wrapping in Taichi GUI for now
            window.GUI.begin('Help', 0.35 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[0]), 0.01, 0.6, 0.30 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[1]))
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
        if final.do_simulate:
            final.k.data_step(final.data_deposit, final.current_deposit_index)
            final.k.agent_step(\
                final.sense_distance,\
                final.sense_angle,\
                third.STEERING_RATE,\
                final.sampling_exponent,\
                final.step_size,\
                final.agent_deposit,\
                final.current_deposit_index,\
                third.distance_sampling_distribution,\
                third.directional_sampling_distribution,\
                third.directional_mutation_type,\
                third.deposit_fetching_strategy,\
                third.agent_boundary_handling)
            final.k.deposit_relaxation_step(final.deposit_attenuation, final.current_deposit_index)
            final.k.trace_relaxation_step(final.trace_attenuation)
            final.current_deposit_index = 1 - final.current_deposit_index
    
        ## Render visualization
        final.k.render_visualization(final.deposit_vis, final.trace_vis, final.current_deposit_index)
        canvas.set_image(fourth.vis_field)
    
        if do_screenshot:
            window.write_image(fourth.ROOT + 'capture/screenshot_' + final.stamp() + '.png') ## Must appear before window.show() call
        window.show()
        if do_export:
            final.store_fit()
        if do_quit:
            break
        
    window.destroy()

## Store fits
current_stamp = final.stamp()
deposit = fourth.deposit_field.to_numpy()
np.save(fourth.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
trace = fourth.trace_field.to_numpy()
np.save(fourth.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)

## Plot results
## Compare with stored fields
current_stamp, deposit, trace = final.store_fit()

plt.figure(figsize = (10.0, 10.0))
plt.imshow(np.flip(np.transpose(deposit[:,:,0]), axis=0))
plt.figure(figsize = (10.0, 10.0))
deposit_restored = np.load(fourth.ROOT + 'data/fits/deposit_' + current_stamp + '.npy')
plt.imshow(np.flip(np.transpose(deposit_restored[:,:,0]), axis=0))

plt.figure(figsize = (10.0, 10.0))
plt.imshow(np.flip(np.transpose(trace[:,:,0]), axis=0))
plt.figure(figsize = (10.0, 10.0))
trace_restored = np.load(fourth.ROOT + 'data/fits/trace_' + current_stamp + '.npy')
plt.imshow(np.flip(np.transpose(trace_restored[:,:,0]), axis=0))

