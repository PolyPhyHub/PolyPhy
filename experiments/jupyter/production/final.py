import finalKernels
import third
import fourth
import first
import time
import math, os
from datetime import datetime
import numpy as np

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
    if not os.path.exists(fourth.ROOT + "data/fits/"):
        os.makedirs(fourth.ROOT + "data/fits/")
    current_stamp = stamp()
    deposit = fourth.deposit_field.to_numpy()
    np.save(fourth.ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
    trace = fourth.trace_field.to_numpy()
    np.save(fourth.ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
    return current_stamp, deposit, trace
