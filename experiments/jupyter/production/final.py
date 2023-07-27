import finalKernels
import fourth
import first
import time
import math, os
from datetime import datetime
import numpy as np
from fourth import DataLoader, FieldVariables, Agents, DerivedVariables
from third import SimulationConstants
from first import TypeAliases 
import taichi as ti

class SimulationVisuals:

    def initGPU(k):
        ## Initialize GPU fields
        FieldVariables.data_field.from_numpy(DataLoader.data)
        FieldVariables.agents_field.from_numpy(Agents.agents)
        k.zero_field(FieldVariables.deposit_field)
        k.zero_field(FieldVariables.trace_field)
        k.zero_field(FieldVariables.vis_field)

    ## Insert a new data point, Round-Robin style, and upload to GPU
    ## This can be very costly for many data points! (eg 10^5 or more)
    def edit_data(edit_index: TypeAliases.INT_CPU, window: ti.ui.Window) -> TypeAliases.INT_CPU:
        mouse_rel_pos = window.get_cursor_pos()
        mouse_rel_pos = (np.min([np.max([0.001, window.get_cursor_pos()[0]]), 0.999]), np.min([np.max([0.001, window.get_cursor_pos()[1]]), 0.999]))
        mouse_pos = np.add(DataLoader.DOMAIN_MIN, np.multiply(mouse_rel_pos, DataLoader.DOMAIN_SIZE))
        DataLoader.data[edit_index, :] = mouse_pos[0], mouse_pos[1], DataLoader.AVG_WEIGHT
        FieldVariables.data_field.from_numpy(DataLoader.data)
        edit_index = (edit_index + 1) % DataLoader.N_DATA
        return edit_index

    ## Current timestamp
    def stamp() -> str:
        return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    ## Store current deposit and trace fields
    def store_fit():
        if not os.path.exists(DataLoader.ROOT + "data/fits/"):
            os.makedirs(DataLoader.ROOT + "data/fits/")
        current_stamp = SimulationVisuals.stamp()
        deposit = FieldVariables.deposit_field.to_numpy()
        np.save(DataLoader.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = FieldVariables.trace_field.to_numpy()
        np.save(DataLoader.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
        return current_stamp, deposit, trace

    def __init__(self,k):
        SimulationVisuals.initGPU(k)

    ## Main simulation & vis loop
    sense_distance = 0.005 * DerivedVariables.DOMAIN_SIZE_MAX
    sense_angle = 1.5
    step_size = 0.0005 * DerivedVariables.DOMAIN_SIZE_MAX
    sampling_exponent = 2.0
    deposit_attenuation = 0.9
    trace_attenuation = 0.96
    data_deposit = 0.1 * SimulationConstants.MAX_DEPOSIT
    agent_deposit = data_deposit * DerivedVariables.DATA_TO_AGENTS_RATIO
    deposit_vis = 0.1
    trace_vis = 1.0

    current_deposit_index = 0
    data_edit_index = 0
    do_simulate = True
    hide_UI = False