import taichi as ti
import taichi.math as timath
import first
import second
import third
import fourth

@ti.data_oriented
class Kernels:
    ## Define all GPU functions and kernels for data and agent processing
    @ti.kernel
    def zero_field(self,f: ti.template()):
        for cell in ti.grouped(f):
            f[cell].fill(0.0)
        return

    @ti.kernel
    def copy_field(self,dst: ti.template(), src: ti.template()): 
        for cell in ti.grouped(dst):
            dst[cell] = src[cell]
        return

    @ti.func
    def world_to_grid_2D(self,pos_world, domain_min, domain_max, grid_resolution) -> first.VEC2i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(grid_resolution, first.FLOAT_GPU), first.INT_GPU)
        return ti.max(first.VEC2i(0, 0), ti.min(grid_coord, grid_resolution - (1, 1)))

    @ti.func
    def angle_to_dir_2D(self,angle) -> first.VEC2f:
        return timath.normalize(first.VEC2f(ti.cos(angle), ti.sin(angle)))

    @ti.func
    def custom_mod(self,a, b) -> first.FLOAT_GPU:
        return a - b * ti.floor(a / b)

    @ti.kernel
    def data_step(self,data_deposit: first.FLOAT_GPU, current_deposit_index: first.INT_GPU):
        for point in ti.ndrange(fourth.data_field.shape[0]):
            pos = first.VEC2f(0.0, 0.0)
            pos[0], pos[1], weight = fourth.data_field[point]
            deposit_cell = self.world_to_grid_2D(pos, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))
            fourth.deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return

    @ti.kernel
    def agent_step(self,sense_distance: first.FLOAT_GPU,\
                sense_angle: first.FLOAT_GPU,\
                STEERING_RATE: first.FLOAT_GPU,\
                sampling_exponent: first.FLOAT_GPU,\
                step_size: first.FLOAT_GPU,\
                agent_deposit: first.FLOAT_GPU,\
                current_deposit_index: first.INT_GPU,\
                distance_sampling_distribution: first.INT_GPU,\
                directional_sampling_distribution: first.INT_GPU,\
                directional_mutation_type: first.INT_GPU,\
                deposit_fetching_strategy: first.INT_GPU,\
                agent_boundary_handling: first.INT_GPU):
        for agent in ti.ndrange(fourth.agents_field.shape[0]):
            pos = first.VEC2f(0.0, 0.0)
            pos[0], pos[1], angle, weight = fourth.agents_field[agent]
            
            ## Generate new mutated angle by perturbing the original
            dir_fwd = self.angle_to_dir_2D(angle)
            angle_mut = angle
            if third.directional_sampling_distribution == second.EnumDirectionalSamplingDistribution.DISCRETE:
                angle_mut += (1.0 if ti.random(dtype=first.FLOAT_GPU) > 0.5 else -1.0) * sense_angle
            elif third.directional_sampling_distribution == second.EnumDirectionalSamplingDistribution.CONE:
                angle_mut += 2.0 * (ti.random(dtype=first.FLOAT_GPU) - 0.5) * sense_angle
            dir_mut = self.angle_to_dir_2D(angle_mut)

            ## Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if third.distance_sampling_distribution == second.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=first.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif third.distance_sampling_distribution == second.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=first.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log( (ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326 )
            agent_sensing_distance *= distance_scaling_factor

            ## Fetch deposit to guide the agent
            deposit_fwd = 1.0
            deposit_mut = 0.0
            if third.deposit_fetching_strategy == second.EnumDepositFetchingStrategy.NN:
                deposit_fwd = fourth.deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_fwd, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))][current_deposit_index]
                deposit_mut = fourth.deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_mut, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))][current_deposit_index]
            elif third.deposit_fetching_strategy == second.EnumDepositFetchingStrategy.NN_PERTURBED:
                ## Fetches the deposit by perturbing the original position by a small delta
                ## This provides cheap stochastic filtering instead of multi-fetch filters
                field_dd = 2.0 * ti.cast(fourth.DOMAIN_SIZE[0], first.FLOAT_GPU) / ti.cast(fourth.DEPOSIT_RESOLUTION[0], first.FLOAT_GPU)
                pos_fwd = pos + agent_sensing_distance * dir_fwd + (field_dd * ti.random(dtype=first.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=first.FLOAT_GPU)))
                deposit_fwd = fourth.deposit_field[self.world_to_grid_2D(pos_fwd, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))][current_deposit_index]
                pos_mut = pos + agent_sensing_distance * dir_mut + (field_dd * ti.random(dtype=first.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=first.FLOAT_GPU)))
                deposit_mut = fourth.deposit_field[self.world_to_grid_2D(pos_mut, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))][current_deposit_index]

            ## Generate new direction for the agent based on the sampled deposit
            angle_new = angle
            if third.directional_mutation_type == second.EnumDirectionalMutationType.DETERMINISTIC:
                angle_new = (third.STEERING_RATE * angle_mut + (1.0-third.STEERING_RATE) * angle) if (deposit_mut > deposit_fwd) else (angle)
            elif third.directional_mutation_type == second.EnumDirectionalMutationType.PROBABILISTIC:
                p_remain = ti.pow(deposit_fwd, sampling_exponent)
                p_mutate = ti.pow(deposit_mut, sampling_exponent)
                mutation_probability = p_mutate / (p_remain + p_mutate)
                angle_new = (third.STEERING_RATE * angle_mut + (1.0-third.STEERING_RATE) * angle) if (ti.random(dtype=first.FLOAT_GPU) < mutation_probability) else (angle)
            dir_new = self.angle_to_dir_2D(angle_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            ## Agent behavior at domain boundaries
            if third.agent_boundary_handling == second.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(pos_new[0] - fourth.DOMAIN_MIN[0] + fourth.DOMAIN_SIZE[0], fourth.DOMAIN_SIZE[0]) + fourth.DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(pos_new[1] - fourth.DOMAIN_MIN[1] + fourth.DOMAIN_SIZE[1], fourth.DOMAIN_SIZE[1]) + fourth.DOMAIN_MIN[1]
            elif third.agent_boundary_handling == second.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= fourth.DOMAIN_MIN[0] or pos_new[0] >= fourth.DOMAIN_MAX[0] or pos_new[1] <= fourth.DOMAIN_MIN[1] or pos_new[1] >= fourth.DOMAIN_MAX[1]:
                    pos_new[0] = 0.5 * (fourth.DOMAIN_MIN[0] + fourth.DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (fourth.DOMAIN_MIN[1] + fourth.DOMAIN_MAX[1])
            elif third.agent_boundary_handling == second.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= fourth.DOMAIN_MIN[0] or pos_new[0] >= fourth.DOMAIN_MAX[0] or pos_new[1] <= fourth.DOMAIN_MIN[1] or pos_new[1] >= fourth.DOMAIN_MAX[1]:
                    pos_new[0] = fourth.DOMAIN_MIN[0] + timath.clamp(ti.random(dtype=first.FLOAT_GPU), 0.001, 0.999) * fourth.DOMAIN_SIZE[0]
                    pos_new[1] = fourth.DOMAIN_MIN[1] + timath.clamp(ti.random(dtype=first.FLOAT_GPU), 0.001, 0.999) * fourth.DOMAIN_SIZE[1]

            fourth.agents_field[agent][0] = pos_new[0]
            fourth.agents_field[agent][1] = pos_new[1]
            fourth.agents_field[agent][2] = angle_new

            ## Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_2D(pos_new, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.DEPOSIT_RESOLUTION))
            fourth.deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_2D(pos_new, first.VEC2f(fourth.DOMAIN_MIN), first.VEC2f(fourth.DOMAIN_MAX), first.VEC2i(fourth.TRACE_RESOLUTION))
            fourth.trace_field[trace_cell][0] += ti.max(1.0e-4, ti.cast(fourth.N_DATA, first.FLOAT_GPU) / ti.cast(fourth.N_AGENTS, first.FLOAT_GPU)) * weight
        return

    DIFFUSION_KERNEL = [1.0, 1.0, 0.707]
    DIFFUSION_KERNEL_NORM = DIFFUSION_KERNEL[0] + 4.0 * DIFFUSION_KERNEL[1] + 4.0 * DIFFUSION_KERNEL[2]

    @ti.kernel
    def deposit_relaxation_step(self,attenuation: first.FLOAT_GPU, current_deposit_index: first.INT_GPU):
        for cell in ti.grouped(fourth.deposit_field):
            ## The "beautiful" expression below implements a 3x3 kernel diffusion with manually wrapped addressing
            ## Taichi doesn't support modulo for tuples so each dimension is handled separately
            value =   self.DIFFUSION_KERNEL[0] * fourth.deposit_field[( (cell[0] + 0 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * fourth.deposit_field[( (cell[0] - 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * fourth.deposit_field[( (cell[0] + 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * fourth.deposit_field[( (cell[0] + 0 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * fourth.deposit_field[( (cell[0] + 0 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * fourth.deposit_field[( (cell[0] - 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * fourth.deposit_field[( (cell[0] + 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * fourth.deposit_field[( (cell[0] + 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * fourth.deposit_field[( (cell[0] - 1 + fourth.DEPOSIT_RESOLUTION[0]) % fourth.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + fourth.DEPOSIT_RESOLUTION[1]) % fourth.DEPOSIT_RESOLUTION[1])][current_deposit_index]
            fourth.deposit_field[cell][1 - current_deposit_index] = attenuation * value / self.DIFFUSION_KERNEL_NORM
        return

    @ti.kernel
    def trace_relaxation_step(self,attenuation: first.FLOAT_GPU):
        for cell in ti.grouped(fourth.trace_field):
            ## Perturb the attenuation by a small factor to avoid accumulating quantization errors
            fourth.trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(dtype=first.FLOAT_GPU)
        return

    @ti.kernel
    def render_visualization(self,deposit_vis: first.FLOAT_GPU, trace_vis: first.FLOAT_GPU, current_deposit_index: first.INT_GPU):
        for x, y in ti.ndrange(fourth.vis_field.shape[0], fourth.vis_field.shape[1]):
            deposit_val = fourth.deposit_field[x * fourth.DEPOSIT_RESOLUTION[0] // fourth.VIS_RESOLUTION[0], y * fourth.DEPOSIT_RESOLUTION[1] // fourth.VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = fourth.trace_field[x * fourth.TRACE_RESOLUTION[0] // fourth.VIS_RESOLUTION[0], y * fourth.TRACE_RESOLUTION[1] // fourth.VIS_RESOLUTION[1]]
            fourth.vis_field[x, y] = ti.pow(first.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return
