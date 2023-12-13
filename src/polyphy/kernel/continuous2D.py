# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath

from core.common import PPTypes, PPConfig
from .common import PPKernels


@ti.data_oriented
class PPKernels_2DContinuous(PPKernels):
    
    @ti.kernel
    def data_step_2D_continuous(
                self,
                data_deposit: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                DOMAIN_MIN: PPTypes.VEC2f,
                DOMAIN_MAX: PPTypes.VEC2f,
                DOMAIN_SIZE: PPTypes.VEC2f,
                DATA_RESOLUTION: PPTypes.VEC2i,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                data_field: ti.template(),
                deposit_field: ti.template()):
        for cell in ti.grouped(deposit_field):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos = PPTypes.VEC2f(DOMAIN_SIZE) * ti.cast(cell, PPTypes.FLOAT_GPU) / PPTypes.VEC2f(DEPOSIT_RESOLUTION)
            data_val = data_field[self.world_to_grid_2D(pos, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DATA_RESOLUTION))][0]
            deposit_field[cell][current_deposit_index] += data_deposit * data_val
        return

    @ti.kernel
    def agent_step_2D_continuous(
                self,
                sense_distance: PPTypes.FLOAT_GPU,
                sense_angle: PPTypes.FLOAT_GPU,
                steering_rate: PPTypes.FLOAT_GPU,
                sampling_exponent: PPTypes.FLOAT_GPU,
                step_size: PPTypes.FLOAT_GPU,
                agent_deposit: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                distance_sampling_distribution: PPTypes.INT_GPU,
                directional_sampling_distribution: PPTypes.INT_GPU,
                directional_mutation_type: PPTypes.INT_GPU,
                deposit_fetching_strategy: PPTypes.INT_GPU,
                agent_boundary_handling: PPTypes.INT_GPU,
                N_DATA: PPTypes.FLOAT_GPU,
                N_AGENTS: PPTypes.FLOAT_GPU,
                DOMAIN_SIZE: PPTypes.VEC2f,
                DOMAIN_MIN: PPTypes.VEC2f,
                DOMAIN_MAX: PPTypes.VEC2f,
                TRACE_RESOLUTION: PPTypes.VEC2i,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                agents_field: ti.template(),
                trace_field: ti.template(),
                deposit_field: ti.template()):
        for agent in ti.ndrange(agents_field.shape[0]):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos[0], pos[1], angle, weight = agents_field[agent]

            # Generate new mutated direction by perturbing the original
            dir_fwd = self.angle_to_dir_2D(angle)
            angle_mut = angle
            if directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.DISCRETE:
                angle_mut += (1.0 if ti.random(dtype=PPTypes.FLOAT_GPU) > 0.5 else -1.0) * sense_angle
            elif directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.CONE:
                angle_mut += 2.0 * (ti.random(dtype=PPTypes.FLOAT_GPU) - 0.5) * sense_angle
            dir_mut = self.angle_to_dir_2D(angle_mut)

            # Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999)
                # log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999)
                # log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log((ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326)
            agent_sensing_distance *= distance_scaling_factor

            # Fetch deposit to guide the agent
            deposit_fwd = 1.0
            deposit_mut = 0.0
            if deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN:
                deposit_fwd = deposit_field[self.world_to_grid_2D(
                    pos + agent_sensing_distance * dir_fwd,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                deposit_mut = deposit_field[self.world_to_grid_2D(
                    pos + agent_sensing_distance * dir_mut,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
            elif deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN_PERTURBED:
                # Fetches the deposit by perturbing the original position by small delta
                # Provides cheap stochastic filtering instead of multi-fetch filters
                field_dd = 2.0 * ti.cast(DOMAIN_SIZE[0], PPTypes.FLOAT_GPU) / ti.cast(DEPOSIT_RESOLUTION[0], PPTypes.FLOAT_GPU)
                pos_fwd = pos + agent_sensing_distance * dir_fwd + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_fwd = deposit_field[self.world_to_grid_2D(
                    pos_fwd, PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                pos_mut = pos + agent_sensing_distance * dir_mut + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_mut = deposit_field[self.world_to_grid_2D(
                    pos_mut,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]

            # Generate new direction for the agent based on the sampled deposit
            angle_new = angle
            if directional_mutation_type == PPConfig.EnumDirectionalMutationType.DETERMINISTIC:
                angle_new = (steering_rate * angle_mut + (1.0-steering_rate) * angle) if (deposit_mut > deposit_fwd) else (angle)
            elif directional_mutation_type == PPConfig.EnumDirectionalMutationType.PROBABILISTIC:
                p_remain = ti.pow(deposit_fwd, sampling_exponent)
                p_mutate = ti.pow(deposit_mut, sampling_exponent)
                mutation_probability = p_mutate / (p_remain + p_mutate)
                angle_new = (steering_rate * angle_mut + (1.0-steering_rate) * angle) if (ti.random(dtype=PPTypes.FLOAT_GPU) < mutation_probability) else (angle)
            dir_new = self.angle_to_dir_2D(angle_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            # Agent behavior at domain boundaries
            if agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(
                    pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0],
                    DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(
                    pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1],
                    DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                        or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                        or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = DOMAIN_MIN[0] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[0]
                    pos_new[1] = DOMAIN_MIN[1] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[1]

            agents_field[agent][0] = pos_new[0]
            agents_field[agent][1] = pos_new[1]
            agents_field[agent][2] = angle_new

            # Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_2D(pos_new,
                                                 PPTypes.VEC2f(DOMAIN_MIN),
                                                 PPTypes.VEC2f(DOMAIN_MAX),
                                                 PPTypes.VEC2i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_2D(
                pos_new, PPTypes.VEC2f(DOMAIN_MIN),
                PPTypes.VEC2f(DOMAIN_MAX),
                PPTypes.VEC2i(TRACE_RESOLUTION))
            trace_field[trace_cell][0] += ti.max(
                1.0e-4,
                ti.cast(
                    N_DATA,
                    PPTypes.FLOAT_GPU) / ti.cast(
                        N_AGENTS,
                        PPTypes.FLOAT_GPU)) * weight
        return
