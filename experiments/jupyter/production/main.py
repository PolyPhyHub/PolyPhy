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
import kernels
import finalKernels

k = finalKernels.FinalKernels()

## Initialize GPU fields
fourth.data_field.from_numpy(fourth.data)
fourth.agents_field.from_numpy(fourth.agents)
k.zero_field(fourth.deposit_field)
k.zero_field(fourth.trace_field)
k.zero_field(fourth.vis_field)

## Main simulation & vis loop
sense_distance = 0.005 * fourth.DOMAIN_SIZE_MAX
sense_angle = 1.5
step_size = 0.0005 * fourth.DOMAIN_SIZE_MAX
sampling_exponent = 2.0
deposit_attenuation = 0.9
trace_attenuation = 0.96
data_deposit = 0.1 * third.MAX_DEPOSIT
agent_deposit = data_deposit * fourth.DATA_TO_AGENTS_RATIO
deposit_vis = 0.1
trace_vis = 1.0

current_deposit_index = 0
data_edit_index = 0
do_simulate = True
hide_UI = False

## Insert a new data point, Round-Robin style, and upload to GPU
## This can be very costly for many data points! (eg 10^5 or more)
def edit_data(edit_index: first.INT_CPU) -> first.INT_CPU:
    mouse_rel_pos = window.get_cursor_pos()
    mouse_rel_pos = (np.min([np.max([0.001, window.get_cursor_pos()[0]]), 0.999]), np.min([np.max([0.001, window.get_cursor_pos()[1]]), 0.999]))
    mouse_pos = np.add(fourth.DOMAIN_MIN, np.multiply(mouse_rel_pos, fourth.DOMAIN_SIZE))
    data[edit_index, :] = mouse_pos[0], mouse_pos[1], AVG_WEIGHT
    data_field.from_numpy(data)
    edit_index = (edit_index + 1) % fourth.N_DATA
    return edit_index

## Current timestamp
def stamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

## Store current deposit and trace fields
def store_fit():
    if not os.path.exists(ROOT + "data/fits/"):
        os.makedirs(ROOT + "data/fits/")
    current_stamp = stamp()
    deposit = fourth.deposit_field.to_numpy()
    np.save(ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
    trace = fourth.trace_field.to_numpy()
    np.save(ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
    return current_stamp, deposit, trace

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
                data_edit_index = edit_data(data_edit_index)
        if window.is_pressed(ti.ui.RMB):
            data_edit_index = edit_data(data_edit_index)
        
        if not hide_UI:
            ## Draw main interactive control GUI
            window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[0]), 0.74 * 1024.0 / first.FLOAT_CPU(fourth.VIS_RESOLUTION[1]))
            window.GUI.text("MCPM parameters:")
            sense_distance = window.GUI.slider_float('Sensing dist', sense_distance, 0.1, 0.05 * np.max([fourth.DOMAIN_SIZE[0], fourth.DOMAIN_SIZE[1]]))
            sense_angle = window.GUI.slider_float('Sensing angle', sense_angle, 0.01, 0.5 * np.pi)
            sampling_exponent = window.GUI.slider_float('Sampling expo', sampling_exponent, 1.0, 10.0)
            step_size = window.GUI.slider_float('Step size', step_size, 0.0, 0.005 * np.max([fourth.DOMAIN_SIZE[0], fourth.DOMAIN_SIZE[1]]))
            data_deposit = window.GUI.slider_float('Data deposit', data_deposit, 0.0, third.MAX_DEPOSIT)
            agent_deposit = window.GUI.slider_float('Agent deposit', agent_deposit, 0.0, 10.0 * third.MAX_DEPOSIT * fourth.DATA_TO_AGENTS_RATIO)
            deposit_attenuation = window.GUI.slider_float('Deposit attn', deposit_attenuation, 0.8, 0.999)
            trace_attenuation = window.GUI.slider_float('Trace attn', trace_attenuation, 0.8, 0.999)
            deposit_vis = math.pow(10.0, window.GUI.slider_float('Deposit vis', math.log(deposit_vis, 10.0), -3.0, 3.0))
            trace_vis = math.pow(10.0, window.GUI.slider_float('Trace vis', math.log(trace_vis, 10.0), -3.0, 3.0))
    
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
            do_simulate = window.GUI.checkbox("Run simulation", do_simulate)
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
        if do_simulate:
            k.data_step(data_deposit, current_deposit_index)
            k.agent_step(\
                sense_distance,\
                sense_angle,\
                third.STEERING_RATE,\
                sampling_exponent,\
                step_size,\
                agent_deposit,\
                current_deposit_index,\
                third.distance_sampling_distribution,\
                third.directional_sampling_distribution,\
                third.directional_mutation_type,\
                third.deposit_fetching_strategy,\
                third.agent_boundary_handling)
            k.deposit_relaxation_step(deposit_attenuation, current_deposit_index)
            k.trace_relaxation_step(trace_attenuation)
            current_deposit_index = 1 - current_deposit_index
    
        ## Render visualization
        k.render_visualization(deposit_vis, trace_vis, current_deposit_index)
        canvas.set_image(fourth.vis_field)
    
        if do_screenshot:
            window.write_image(ROOT + 'capture/screenshot_' + stamp() + '.png') ## Must appear before window.show() call
        window.show()
        if do_export:
            store_fit()
        if do_quit:
            break
        
    window.destroy()

## Store fits
current_stamp = stamp()
deposit = fourth.deposit_field.to_numpy()
np.save(ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
trace = fourth.trace_field.to_numpy()
np.save(ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)

## Plot results
## Compare with stored fields
current_stamp, deposit, trace = store_fit()

plt.figure(figsize = (10.0, 10.0))
plt.imshow(np.flip(np.transpose(deposit[:,:,0]), axis=0))
plt.figure(figsize = (10.0, 10.0))
deposit_restored = np.load(ROOT + 'data/fits/deposit_' + current_stamp + '.npy')
plt.imshow(np.flip(np.transpose(deposit_restored[:,:,0]), axis=0))

plt.figure(figsize = (10.0, 10.0))
plt.imshow(np.flip(np.transpose(trace[:,:,0]), axis=0))
plt.figure(figsize = (10.0, 10.0))
trace_restored = np.load(ROOT + 'data/fits/trace_' + current_stamp + '.npy')
plt.imshow(np.flip(np.transpose(trace_restored[:,:,0]), axis=0))

