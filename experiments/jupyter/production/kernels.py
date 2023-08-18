import taichi as ti
import taichi.math as timath
from polyphy_functions import TypeAliases, StateFlags, SimulationConstants, StateFlags

@ti.data_oriented
class Kernels:
    def __init__(self,derivedVariables,fieldVariables, dataLoaders):
        self.derivedVariables = derivedVariables
        self.fieldVariables = fieldVariables
        self.dataLoaders = dataLoaders
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
    def world_to_grid_2D(self,pos_world, domain_min, domain_max, grid_resolution) -> TypeAliases.VEC2i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(grid_resolution, TypeAliases.FLOAT_GPU), TypeAliases.INT_GPU)
        return ti.max(TypeAliases.VEC2i(0, 0), ti.min(grid_coord, grid_resolution - (1, 1)))

    @ti.func
    def angle_to_dir_2D(self,angle) -> TypeAliases.VEC2f:
        return timath.normalize(TypeAliases.VEC2f(ti.cos(angle), ti.sin(angle)))

    @ti.func
    def custom_mod(self,a, b) -> TypeAliases.FLOAT_GPU:
        return a - b * ti.floor(a / b)

    @ti.kernel
    def data_step(self,data_deposit: TypeAliases.FLOAT_GPU, current_deposit_index: TypeAliases.INT_GPU):
        for point in ti.ndrange(self.fieldVariables.data_field.shape[0]):
            pos = TypeAliases.VEC2f(0.0, 0.0)
            pos[0], pos[1], weight = self.fieldVariables.data_field[point]
            deposit_cell = self.world_to_grid_2D(pos, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))
            self.fieldVariables.deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return

    @ti.kernel
    def agent_step(self,sense_distance: TypeAliases.FLOAT_GPU,\
                sense_angle: TypeAliases.FLOAT_GPU,\
                STEERING_RATE: TypeAliases.FLOAT_GPU,\
                sampling_exponent: TypeAliases.FLOAT_GPU,\
                step_size: TypeAliases.FLOAT_GPU,\
                agent_deposit: TypeAliases.FLOAT_GPU,\
                current_deposit_index: TypeAliases.INT_GPU,\
                distance_sampling_distribution: TypeAliases.INT_GPU,\
                directional_sampling_distribution: TypeAliases.INT_GPU,\
                directional_mutation_type: TypeAliases.INT_GPU,\
                deposit_fetching_strategy: TypeAliases.INT_GPU,\
                agent_boundary_handling: TypeAliases.INT_GPU):
        for agent in ti.ndrange(self.fieldVariables.agents_field.shape[0]):
            pos = TypeAliases.VEC2f(0.0, 0.0)
            pos[0], pos[1], angle, weight = self.fieldVariables.agents_field[agent]
            
            ## Generate new mutated angle by perturbing the original
            dir_fwd = self.angle_to_dir_2D(angle)
            angle_mut = angle
            if StateFlags.directional_sampling_distribution == StateFlags.EnumDirectionalSamplingDistribution.DISCRETE:
                angle_mut += (1.0 if ti.random(dtype=TypeAliases.FLOAT_GPU) > 0.5 else -1.0) * sense_angle
            elif StateFlags.directional_sampling_distribution == StateFlags.EnumDirectionalSamplingDistribution.CONE:
                angle_mut += 2.0 * (ti.random(dtype=TypeAliases.FLOAT_GPU) - 0.5) * sense_angle
            dir_mut = self.angle_to_dir_2D(angle_mut)

            ## Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if StateFlags.distance_sampling_distribution == StateFlags.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=TypeAliases.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif StateFlags.distance_sampling_distribution == StateFlags.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=TypeAliases.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log( (ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326 )
            agent_sensing_distance *= distance_scaling_factor

            ## Fetch deposit to guide the agent
            deposit_fwd = 1.0
            deposit_mut = 0.0
            if StateFlags.deposit_fetching_strategy == StateFlags.EnumDepositFetchingStrategy.NN:
                deposit_fwd = self.fieldVariables.deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_fwd, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))][current_deposit_index]
                deposit_mut = self.fieldVariables.deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_mut, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))][current_deposit_index]
            elif StateFlags.deposit_fetching_strategy == StateFlags.EnumDepositFetchingStrategy.NN_PERTURBED:
                ## Fetches the deposit by perturbing the original position by a small delta
                ## This provides cheap stochastic filtering instead of multi-fetch filters
                field_dd = 2.0 * ti.cast(self.dataLoaders.DOMAIN_SIZE[0], TypeAliases.FLOAT_GPU) / ti.cast(self.derivedVariables.DEPOSIT_RESOLUTION[0], TypeAliases.FLOAT_GPU)
                pos_fwd = pos + agent_sensing_distance * dir_fwd + (field_dd * ti.random(dtype=TypeAliases.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=TypeAliases.FLOAT_GPU)))
                deposit_fwd = self.fieldVariables.deposit_field[self.world_to_grid_2D(pos_fwd, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))][current_deposit_index]
                pos_mut = pos + agent_sensing_distance * dir_mut + (field_dd * ti.random(dtype=TypeAliases.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=TypeAliases.FLOAT_GPU)))
                deposit_mut = self.fieldVariables.deposit_field[self.world_to_grid_2D(pos_mut, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))][current_deposit_index]

            ## Generate new direction for the agent based on the sampled deposit
            angle_new = angle
            if StateFlags.directional_mutation_type == StateFlags.EnumDirectionalMutationType.DETERMINISTIC:
                angle_new = (SimulationConstants.STEERING_RATE * angle_mut + (1.0-SimulationConstants.STEERING_RATE) * angle) if (deposit_mut > deposit_fwd) else (angle)
            elif StateFlags.directional_mutation_type == StateFlags.EnumDirectionalMutationType.PROBABILISTIC:
                p_remain = ti.pow(deposit_fwd, sampling_exponent)
                p_mutate = ti.pow(deposit_mut, sampling_exponent)
                mutation_probability = p_mutate / (p_remain + p_mutate)
                angle_new = (SimulationConstants.STEERING_RATE * angle_mut + (1.0-SimulationConstants.STEERING_RATE) * angle) if (ti.random(dtype=TypeAliases.FLOAT_GPU) < mutation_probability) else (angle)
            dir_new = self.angle_to_dir_2D(angle_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            ## Agent behavior at domain boundaries
            if StateFlags.agent_boundary_handling == StateFlags.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(pos_new[0] - self.dataLoaders.DOMAIN_MIN[0] + self.dataLoaders.DOMAIN_SIZE[0], self.dataLoaders.DOMAIN_SIZE[0]) + self.dataLoaders.DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(pos_new[1] - self.dataLoaders.DOMAIN_MIN[1] + self.dataLoaders.DOMAIN_SIZE[1], self.dataLoaders.DOMAIN_SIZE[1]) + self.dataLoaders.DOMAIN_MIN[1]
            elif StateFlags.agent_boundary_handling == StateFlags.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= self.dataLoaders.DOMAIN_MIN[0] or pos_new[0] >= self.dataLoaders.DOMAIN_MAX[0] or pos_new[1] <= self.dataLoaders.DOMAIN_MIN[1] or pos_new[1] >= self.dataLoaders.DOMAIN_MAX[1]:
                    pos_new[0] = 0.5 * (self.dataLoaders.DOMAIN_MIN[0] + self.dataLoaders.DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (self.dataLoaders.DOMAIN_MIN[1] + self.dataLoaders.DOMAIN_MAX[1])
            elif StateFlags.agent_boundary_handling == StateFlags.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= self.dataLoaders.DOMAIN_MIN[0] or pos_new[0] >= self.dataLoaders.DOMAIN_MAX[0] or pos_new[1] <= self.dataLoaders.DOMAIN_MIN[1] or pos_new[1] >= self.dataLoaders.DOMAIN_MAX[1]:
                    pos_new[0] = self.dataLoaders.DOMAIN_MIN[0] + timath.clamp(ti.random(dtype=TypeAliases.FLOAT_GPU), 0.001, 0.999) * self.dataLoaders.DOMAIN_SIZE[0]
                    pos_new[1] = self.dataLoaders.DOMAIN_MIN[1] + timath.clamp(ti.random(dtype=TypeAliases.FLOAT_GPU), 0.001, 0.999) * self.dataLoaders.DOMAIN_SIZE[1]

            self.fieldVariables.agents_field[agent][0] = pos_new[0]
            self.fieldVariables.agents_field[agent][1] = pos_new[1]
            self.fieldVariables.agents_field[agent][2] = angle_new

            ## Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_2D(pos_new, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.DEPOSIT_RESOLUTION))
            self.fieldVariables.deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_2D(pos_new, TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MIN), TypeAliases.VEC2f(self.dataLoaders.DOMAIN_MAX), TypeAliases.VEC2i(self.derivedVariables.TRACE_RESOLUTION))
            self.fieldVariables.trace_field[trace_cell][0] += ti.max(1.0e-4, ti.cast(self.dataLoaders.N_DATA, TypeAliases.FLOAT_GPU) / ti.cast(self.dataLoaders.N_AGENTS, TypeAliases.FLOAT_GPU)) * weight
        return

    DIFFUSION_KERNEL = [1.0, 1.0, 0.707]
    DIFFUSION_KERNEL_NORM = DIFFUSION_KERNEL[0] + 4.0 * DIFFUSION_KERNEL[1] + 4.0 * DIFFUSION_KERNEL[2]

    @ti.kernel
    def deposit_relaxation_step(self,attenuation: TypeAliases.FLOAT_GPU, current_deposit_index: TypeAliases.INT_GPU):
        for cell in ti.grouped(self.fieldVariables.deposit_field):
            ## The "beautiful" expression below implements a 3x3 kernel diffusion with manually wrapped addressing
            ## Taichi doesn't support modulo for tuples so each dimension is handled separately
            value =   self.DIFFUSION_KERNEL[0] * self.fieldVariables.deposit_field[( (cell[0] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * self.fieldVariables.deposit_field[( (cell[0] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * self.fieldVariables.deposit_field[( (cell[0] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * self.fieldVariables.deposit_field[( (cell[0] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[1] * self.fieldVariables.deposit_field[( (cell[0] + 0 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * self.fieldVariables.deposit_field[( (cell[0] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * self.fieldVariables.deposit_field[( (cell[0] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * self.fieldVariables.deposit_field[( (cell[0] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + self.DIFFUSION_KERNEL[2] * self.fieldVariables.deposit_field[( (cell[0] - 1 + self.derivedVariables.DEPOSIT_RESOLUTION[0]) % self.derivedVariables.DEPOSIT_RESOLUTION[0], (cell[1] + 1 + self.derivedVariables.DEPOSIT_RESOLUTION[1]) % self.derivedVariables.DEPOSIT_RESOLUTION[1])][current_deposit_index]
            self.fieldVariables.deposit_field[cell][1 - current_deposit_index] = attenuation * value / self.DIFFUSION_KERNEL_NORM
        return

    @ti.kernel
    def trace_relaxation_step(self,attenuation: TypeAliases.FLOAT_GPU):
        for cell in ti.grouped(self.fieldVariables.trace_field):
            ## Perturb the attenuation by a small factor to avoid accumulating quantization errors
            self.fieldVariables.trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(dtype=TypeAliases.FLOAT_GPU)
        return

    @ti.kernel
    def render_visualization(self,deposit_vis: TypeAliases.FLOAT_GPU, trace_vis: TypeAliases.FLOAT_GPU, current_deposit_index: TypeAliases.INT_GPU):
        for x, y in ti.ndrange(self.fieldVariables.vis_field.shape[0], self.fieldVariables.vis_field.shape[1]):
            deposit_val = self.fieldVariables.deposit_field[x * self.derivedVariables.DEPOSIT_RESOLUTION[0] // self.derivedVariables.VIS_RESOLUTION[0], y * self.derivedVariables.DEPOSIT_RESOLUTION[1] // self.derivedVariables.VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = self.fieldVariables.trace_field[x * self.derivedVariables.TRACE_RESOLUTION[0] // self.derivedVariables.VIS_RESOLUTION[0], y * self.derivedVariables.TRACE_RESOLUTION[1] // self.derivedVariables.VIS_RESOLUTION[1]]
            self.fieldVariables.vis_field[x, y] = ti.pow(TypeAliases.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return
