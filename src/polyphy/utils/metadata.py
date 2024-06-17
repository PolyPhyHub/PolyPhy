import json
import os

def get_metadata(ppConfig, ppInternalData):
    metadata = {
        "_dataset": ppConfig.ppData.input_file,
        "number_of_data_points": ppConfig.ppData.data.size,
        "number_of_agents": ppConfig.ppData.N_AGENTS,
        "simulation_grid_resolution": f"{ppConfig.TRACE_RESOLUTION[0]} x {ppConfig.TRACE_RESOLUTION[1]} [vox]",
        "simulation_grid_size": f"{ppConfig.ppData.DOMAIN_SIZE[0]} x {ppConfig.ppData.DOMAIN_SIZE[1]} [mpc]",
        "simulation_grid_center": f"({ppConfig.ppData.DOMAIN_SIZE[0] / 2}, {ppConfig.ppData.DOMAIN_SIZE[1] / 2}) [mpc]",
        "move_distance": ppConfig.step_size,
        "move_distance_grid": ppConfig.step_size / ppConfig.ppData.DOMAIN_SIZE[0] * ppConfig.TRACE_RESOLUTION[0],
        "sense_distance": ppConfig.sense_distance,
        "sense_distance_grid": ppConfig.sense_distance / ppConfig.ppData.DOMAIN_SIZE[0] * ppConfig.TRACE_RESOLUTION[0],
        "move_spread": ppConfig.steering_rate,
        "sense_spread": ppConfig.sense_angle,
        "persistence_coefficient": ppConfig.sampling_exponent,
        "agent_deposit": ppConfig.agent_deposit,
        "sampling_sharpness": ppConfig.data_deposit
    }
    return metadata

def save_metadata(metadata, filename='metadata.json'):
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)
