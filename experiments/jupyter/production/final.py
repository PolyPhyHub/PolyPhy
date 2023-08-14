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
    def initGPU(self,k):
        ## Initialize GPU fields
        self.fieldVariables.data_field.from_numpy(self.dataLoaders.data)
        self.fieldVariables.agents_field.from_numpy(self.agents.agents)
        k.zero_field(self.fieldVariables.deposit_field)
        k.zero_field(self.fieldVariables.trace_field)
        k.zero_field(self.fieldVariables.vis_field)

    ## Insert a new data point, Round-Robin style, and upload to GPU
    ## This can be very costly for many data points! (eg 10^5 or more)
    def edit_data(self,edit_index: TypeAliases.INT_CPU, window: ti.ui.Window) -> TypeAliases.INT_CPU:
        mouse_rel_pos = window.get_cursor_pos()
        mouse_rel_pos = (np.min([np.max([0.001, window.get_cursor_pos()[0]]), 0.999]), np.min([np.max([0.001, window.get_cursor_pos()[1]]), 0.999]))
        mouse_pos = np.add(self.dataLoaders.DOMAIN_MIN, np.multiply(mouse_rel_pos, self.dataLoaders.DOMAIN_SIZE))
        self.dataLoaders.data[edit_index, :] = mouse_pos[0], mouse_pos[1], self.dataLoaders.AVG_WEIGHT
        self.fieldVariables.data_field.from_numpy(self.dataLoaders.data)
        edit_index = (edit_index + 1) % self.dataLoaders.N_DATA
        return edit_index

    ## Current timestamp
    def stamp(self) -> str:
        return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    ## Store current deposit and trace fields
    def store_fit(self):
        if not os.path.exists(self.dataLoaders.ROOT + "data/fits/"):
            os.makedirs(self.dataLoaders.ROOT + "data/fits/")
        current_stamp = self.stamp()
        deposit = self.fieldVariables.deposit_field.to_numpy()
        np.save(self.dataLoaders.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
        trace = self.fieldVariables.trace_field.to_numpy()
        np.save(self.dataLoaders.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
        return current_stamp, deposit, trace

    def __init__(self,k,dataLoaders,derivedVariables, agents, fieldVariables):
        self.dataLoaders = dataLoaders
        self.derivedVariables = derivedVariables
        self.agents = agents
        self.fieldVariables = fieldVariables
        
        self.initGPU(k)
        
        ## Main simulation & vis loop
        self.sense_distance = 0.005 * self.derivedVariables.DOMAIN_SIZE_MAX
        self.sense_angle = 1.5
        self.step_size = 0.0005 * self.derivedVariables.DOMAIN_SIZE_MAX
        self.sampling_exponent = 2.0
        self.deposit_attenuation = 0.9
        self.trace_attenuation = 0.96
        self.data_deposit = 0.1 * SimulationConstants.MAX_DEPOSIT
        self.agent_deposit = self.data_deposit * self.derivedVariables.DATA_TO_AGENTS_RATIO
        self.deposit_vis = 0.1
        self.trace_vis = 1.0

        self.current_deposit_index = 0
        self.data_edit_index = 0
        self.do_simulate = True
        self.hide_UI = False