import taichi as ti
import taichi.math as timath
from polyphy_core import PPTypes, PPConfig

@ti.data_oriented
class PPKernels:

    ## GPU functions (callable by kernels) ====================================================
    @ti.func
    def custom_mod(self,a, b) -> PPTypes.FLOAT_GPU:
        return a - b * ti.floor(a / b)

    @ti.func
    def world_to_grid_2D(self, pos_world, domain_min, domain_max, grid_resolution) -> PPTypes.VEC2i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(grid_resolution, PPTypes.FLOAT_GPU), PPTypes.INT_GPU)
        return ti.max(PPTypes.VEC2i(0, 0), ti.min(grid_coord, grid_resolution - (1, 1)))
    
    @ti.func
    def world_to_grid_3D(self, pos_world, domain_min, domain_max, grid_resolution) -> PPTypes.VEC3i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(grid_resolution, PPTypes.FLOAT_GPU), PPTypes.INT_GPU)
        return ti.max(PPTypes.VEC3i(0, 0, 0), ti.min(grid_coord, grid_resolution - (1, 1, 1)))

    @ti.func
    def angle_to_dir_2D(self, angle) -> PPTypes.VEC2f:
        return timath.normalize(PPTypes.VEC2f(ti.cos(angle), ti.sin(angle)))
    
    @ti.func
    def angles_to_dir_3D(self, theta, phi) -> PPTypes.VEC3f:
        return timath.normalize(PPTypes.VEC3f(ti.sin(theta) * ti.cos(phi), ti.cos(theta), ti.sin(theta) * ti.sin(phi)))

    @ti.func
    def dir_3D_to_angles(self, dir):
        theta = timath.acos(dir[1] / timath.length(dir))
        phi = timath.atan2(dir[2], dir[0])
        return theta, phi

    @ti.func
    def axial_rotate_3D(self, vec, axis, angle):
        return ti.cos(angle) * vec + ti.sin(angle) * (timath.cross(axis, vec)) + timath.dot(axis, vec) * (1.0 - ti.cos(angle)) * axis
    
    @ti.func
    def ray_AABB_intersection(self, ray_pos, ray_dir, AABB_min, AABB_max):
        t0 = (AABB_min[0] - ray_pos[0]) / ray_dir[0]
        t1 = (AABB_max[0] - ray_pos[0]) / ray_dir[0]
        t2 = (AABB_min[1] - ray_pos[1]) / ray_dir[1]
        t3 = (AABB_max[1] - ray_pos[1]) / ray_dir[1]
        t4 = (AABB_min[2] - ray_pos[2]) / ray_dir[2]
        t5 = (AABB_max[2] - ray_pos[2]) / ray_dir[2]
        t6 = ti.max(ti.max(ti.min(t0, t1), ti.min(t2, t3)), ti.min(t4, t5))
        t7 = ti.min(ti.min(ti.max(t0, t1), ti.max(t2, t3)), ti.max(t4, t5))
        return PPTypes.VEC2f(-1.0, -1.0) if (t7 < 0.0 or t6 >= t7) else PPTypes.VEC2f(t6, t7)

    ## GPU kernels (callable by core classes via Taichi API) ====================================
    @ti.kernel
    def zero_field(self, f: ti.template()):
        for cell in ti.grouped(f):
            f[cell].fill(0.0)
        return

    @ti.kernel
    def copy_field(self, dst: ti.template(), src: ti.template()): 
        for cell in ti.grouped(dst):
            dst[cell] = src[cell]
        return
    
    @ti.kernel
    def data_step_2D_discrete(self,\
                data_deposit: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                DOMAIN_MIN: PPTypes.VEC2f,\
                DOMAIN_MAX: PPTypes.VEC2f,\
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,\
                data_field: ti.template(),\
                deposit_field: ti.template()):
        for point in ti.ndrange(data_field.shape[0]):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos[0], pos[1], weight = data_field[point]
            deposit_cell = self.world_to_grid_2D(pos, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return
    
    @ti.kernel
    def data_step_3D_discrete(self,\
                data_deposit: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                DOMAIN_MIN: PPTypes.VEC3f,\
                DOMAIN_MAX: PPTypes.VEC3f,\
                DEPOSIT_RESOLUTION: PPTypes.VEC3i,\
                data_field: ti.template(),\
                deposit_field: ti.template()):
        for point in ti.ndrange(data_field.shape[0]):
            pos = PPTypes.VEC3f(0.0, 0.0, 0.0)
            pos[0], pos[1], pos[2], weight = data_field[point]
            deposit_cell = self.world_to_grid_3D(pos, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return

    @ti.kernel
    def agent_step_2D_discrete(self,\
                sense_distance: PPTypes.FLOAT_GPU,\
                sense_angle: PPTypes.FLOAT_GPU,\
                steering_rate: PPTypes.FLOAT_GPU,\
                sampling_exponent: PPTypes.FLOAT_GPU,\
                step_size: PPTypes.FLOAT_GPU,\
                agent_deposit: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                distance_sampling_distribution: PPTypes.INT_GPU,\
                directional_sampling_distribution: PPTypes.INT_GPU,\
                directional_mutation_type: PPTypes.INT_GPU,\
                deposit_fetching_strategy: PPTypes.INT_GPU,\
                agent_boundary_handling: PPTypes.INT_GPU,
                N_DATA: PPTypes.FLOAT_GPU,\
                N_AGENTS: PPTypes.FLOAT_GPU,\
                DOMAIN_SIZE: PPTypes.VEC2f,\
                DOMAIN_MIN: PPTypes.VEC2f,\
                DOMAIN_MAX: PPTypes.VEC2f,\
                TRACE_RESOLUTION: PPTypes.VEC2i,\
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,\
                agents_field: ti.template(),\
                trace_field: ti.template(),\
                deposit_field: ti.template()):
        for agent in ti.ndrange(agents_field.shape[0]):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos[0], pos[1], angle, weight = agents_field[agent]
            
            ## Generate new mutated direction by perturbing the original
            dir_fwd = self.angle_to_dir_2D(angle)
            angle_mut = angle
            if directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.DISCRETE:
                angle_mut += (1.0 if ti.random(dtype=PPTypes.FLOAT_GPU) > 0.5 else -1.0) * sense_angle
            elif directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.CONE:
                angle_mut += 2.0 * (ti.random(dtype=PPTypes.FLOAT_GPU) - 0.5) * sense_angle
            dir_mut = self.angle_to_dir_2D(angle_mut)

            ## Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log( (ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326 )
            agent_sensing_distance *= distance_scaling_factor

            ## Fetch deposit to guide the agent
            deposit_fwd = 1.0
            deposit_mut = 0.0
            if deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN:
                deposit_fwd = deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_fwd, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                deposit_mut = deposit_field[self.world_to_grid_2D(pos + agent_sensing_distance * dir_mut, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
            elif deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN_PERTURBED:
                ## Fetches the deposit by perturbing the original position by a small delta
                ## This provides cheap stochastic filtering instead of multi-fetch filters
                field_dd = 2.0 * ti.cast(DOMAIN_SIZE[0], PPTypes.FLOAT_GPU) / ti.cast(DEPOSIT_RESOLUTION[0], PPTypes.FLOAT_GPU)
                pos_fwd = pos + agent_sensing_distance * dir_fwd + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_fwd = deposit_field[self.world_to_grid_2D(pos_fwd, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                pos_mut = pos + agent_sensing_distance * dir_mut + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_mut = deposit_field[self.world_to_grid_2D(pos_mut, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]

            ## Generate new direction for the agent based on the sampled deposit
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

            ## Agent behavior at domain boundaries
            if agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0], DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1], DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = DOMAIN_MIN[0] + timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[0]
                    pos_new[1] = DOMAIN_MIN[1] + timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[1]

            agents_field[agent][0] = pos_new[0]
            agents_field[agent][1] = pos_new[1]
            agents_field[agent][2] = angle_new

            ## Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_2D(pos_new, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_2D(pos_new, PPTypes.VEC2f(DOMAIN_MIN), PPTypes.VEC2f(DOMAIN_MAX), PPTypes.VEC2i(TRACE_RESOLUTION))
            trace_field[trace_cell][0] += ti.max(1.0e-4, ti.cast(N_DATA, PPTypes.FLOAT_GPU) / ti.cast(N_AGENTS, PPTypes.FLOAT_GPU)) * weight
        return
    
    @ti.kernel
    def agent_step_3D_discrete(self,\
                sense_distance: PPTypes.FLOAT_GPU,\
                sense_angle: PPTypes.FLOAT_GPU,\
                steering_rate: PPTypes.FLOAT_GPU,\
                sampling_exponent: PPTypes.FLOAT_GPU,\
                step_size: PPTypes.FLOAT_GPU,\
                agent_deposit: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                distance_sampling_distribution: PPTypes.INT_GPU,\
                directional_sampling_distribution: PPTypes.INT_GPU,\
                directional_mutation_type: PPTypes.INT_GPU,\
                deposit_fetching_strategy: PPTypes.INT_GPU,\
                agent_boundary_handling: PPTypes.INT_GPU,
                N_DATA: PPTypes.FLOAT_GPU,\
                N_AGENTS: PPTypes.FLOAT_GPU,\
                DOMAIN_SIZE: PPTypes.VEC3f,\
                DOMAIN_MIN: PPTypes.VEC3f,\
                DOMAIN_MAX: PPTypes.VEC3f,\
                TRACE_RESOLUTION: PPTypes.VEC3i,\
                DEPOSIT_RESOLUTION: PPTypes.VEC3i,\
                agents_field: ti.template(),\
                trace_field: ti.template(),\
                deposit_field: ti.template()):
        for agent in ti.ndrange(agents_field.shape[0]):
            pos = PPTypes.VEC3f(0.0, 0.0, 0.0)
            pos[0], pos[1], pos[2], theta, phi, weight = agents_field[agent]

            ## Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log( (ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326 )
            agent_sensing_distance *= distance_scaling_factor

            ## Generate new mutated direction by perturbing the original
            ## TODO implement the other sampling strategies
            dir_fwd = self.angles_to_dir_3D(theta, phi)
            xi_dir = 1.0
            if directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.CONE:
                xi_dir = ti.random(dtype=PPTypes.FLOAT_GPU)
            theta_sense = theta - xi_dir * sense_angle
            off_fwd_dir = self.angles_to_dir_3D(theta_sense, phi)
            random_azimuth = ti.random(dtype=PPTypes.FLOAT_GPU) * 2.0 * timath.pi - timath.pi
            dir_mut = self.axial_rotate_3D(off_fwd_dir, dir_fwd, random_azimuth)

            ## Fetch deposit to guide the agent
            ## TODO implement the other mutation strategies
            deposit_fwd = deposit_field[self.world_to_grid_3D(pos + agent_sensing_distance * dir_fwd, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
            deposit_mut = deposit_field[self.world_to_grid_3D(pos + agent_sensing_distance * dir_mut, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]

            ## Generate new direction for the agent based on the sampled deposit
            p_remain = ti.pow(deposit_fwd, sampling_exponent)
            p_mutate = ti.pow(deposit_mut, sampling_exponent)
            mutation_probability = p_mutate / (p_remain + p_mutate)
            dir_new = dir_fwd
            theta_new = theta
            phi_new = phi
            if p_remain + p_mutate > 1.0e-5:
                if ti.random(dtype=PPTypes.FLOAT_GPU) < mutation_probability:
                    theta_mut = theta - steering_rate * xi_dir * sense_angle
                    off_mut_dir = self.angles_to_dir_3D(theta_mut, phi)
                    dir_new = self.axial_rotate_3D(off_mut_dir, dir_fwd, random_azimuth)
                    theta_new, phi_new = self.dir_3D_to_angles(dir_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            ## Agent behavior at domain boundaries
            if agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0], DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1], DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
                pos_new[2] = self.custom_mod(pos_new[2] - DOMAIN_MIN[2] + DOMAIN_SIZE[2], DOMAIN_SIZE[2]) + DOMAIN_MIN[2]
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1] or pos_new[2] <= DOMAIN_MIN[2] or pos_new[2] >= DOMAIN_MAX[2]:
                    pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
                    pos_new[2] = 0.5 * (DOMAIN_MIN[2] + DOMAIN_MAX[2])
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1] or pos_new[2] <= DOMAIN_MIN[2] or pos_new[2] >= DOMAIN_MAX[2]:
                    pos_new[0] = DOMAIN_MIN[0] + timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[0]
                    pos_new[1] = DOMAIN_MIN[1] + timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[1]
                    pos_new[2] = DOMAIN_MIN[2] + timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[2]

            agents_field[agent][0] = pos_new[0]
            agents_field[agent][1] = pos_new[1]
            agents_field[agent][2] = pos_new[2]
            agents_field[agent][3] = theta_new
            agents_field[agent][4] = phi_new

            ## Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_3D(pos_new, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_3D(pos_new, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(TRACE_RESOLUTION))
            trace_field[trace_cell][0] += weight
        return

    @ti.kernel
    def deposit_relaxation_step_2D_discrete(self, attenuation: PPTypes.FLOAT_GPU, current_deposit_index: PPTypes.INT_GPU, DEPOSIT_RESOLUTION: PPTypes.VEC2i, deposit_field: ti.template()):
        DIFFUSION_WEIGHTS = [1.0, 1.0, 0.707]
        DIFFUSION_WEIGHTS_NORM = DIFFUSION_WEIGHTS[0] + 4.0 * DIFFUSION_WEIGHTS[1] + 4.0 * DIFFUSION_WEIGHTS[2]
        for cell in ti.grouped(deposit_field):
            ## The "beautiful" expression below implements a 3x3 kernel diffusion with manually wrapped addressing
            ## Taichi doesn't support modulo for tuples so each dimension is handled separately
            value =   DIFFUSION_WEIGHTS[0] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[2] * deposit_field[( (cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[2] * deposit_field[( (cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[2] * deposit_field[( (cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                    + DIFFUSION_WEIGHTS[2] * deposit_field[( (cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]
            deposit_field[cell][1 - current_deposit_index] = attenuation * value / DIFFUSION_WEIGHTS_NORM
        return

    @ti.kernel
    def deposit_relaxation_step_3D_discrete(self, attenuation: PPTypes.FLOAT_GPU, current_deposit_index: PPTypes.INT_GPU, DEPOSIT_RESOLUTION: PPTypes.VEC3i, deposit_field: ti.template()):
        DIFFUSION_WEIGHTS = [1.0, 1.0, 0.0, 0.0]
        DIFFUSION_WEIGHTS_NORM = DIFFUSION_WEIGHTS[0] + 6.0 * DIFFUSION_WEIGHTS[1] + 12.0 * DIFFUSION_WEIGHTS[2] + 8.0 * DIFFUSION_WEIGHTS[3]
        for cell in ti.grouped(deposit_field):
        ## The "beautiful" expression below implements a 3x3x3 kernel diffusion in a 6-neighborhood with manually wrapped addressing
        ## Taichi doesn't support modulo for tuples so each dimension is handled separately
            value = DIFFUSION_WEIGHTS[0] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] - 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]
            deposit_field[cell][1 - current_deposit_index] = attenuation * value / DIFFUSION_WEIGHTS_NORM
        return

    @ti.kernel
    def trace_relaxation_step_2D_discrete(self, attenuation: PPTypes.FLOAT_GPU, trace_field: ti.template()):
        for cell in ti.grouped(trace_field):
            ## Perturb the attenuation by a small factor to avoid accumulating quantization errors
            trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(dtype=PPTypes.FLOAT_GPU)
        return

    @ti.kernel
    def trace_relaxation_step_3D_discrete(self, attenuation: PPTypes.FLOAT_GPU, trace_field: ti.template()):
        for cell in ti.grouped(trace_field):
            ## Perturb the attenuation by a small factor to avoid accumulating quantization errors
            trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(dtype=PPTypes.FLOAT_GPU)
        return

    @ti.kernel
    def render_visualization_2D_discrete(self,\
                trace_vis: PPTypes.FLOAT_GPU,\
                deposit_vis: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                TRACE_RESOLUTION: PPTypes.VEC2i,\
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,\
                VIS_RESOLUTION: PPTypes.VEC2i,\
                trace_field: ti.template(),\
                deposit_field: ti.template(),\
                vis_field: ti.template()):
        for x, y in ti.ndrange(vis_field.shape[0], vis_field.shape[1]):
            deposit_val = deposit_field[x * DEPOSIT_RESOLUTION[0] // VIS_RESOLUTION[0], y * DEPOSIT_RESOLUTION[1] // VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = trace_field[x * TRACE_RESOLUTION[0] // VIS_RESOLUTION[0], y * TRACE_RESOLUTION[1] // VIS_RESOLUTION[1]]
            vis_field[x, y] = ti.pow(PPTypes.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return

    @ti.kernel
    def render_visualization_3D_raymarched(self,\
                trace_vis: PPTypes.FLOAT_GPU,\
                deposit_vis: PPTypes.FLOAT_GPU,\
                camera_distance: PPTypes.FLOAT_GPU,\
                camera_polar: PPTypes.FLOAT_GPU,\
                camera_azimuth: PPTypes.FLOAT_GPU,\
                n_ray_steps_f: PPTypes.FLOAT_GPU,\
                current_deposit_index: PPTypes.INT_GPU,\
                TRACE_RESOLUTION: PPTypes.VEC3i,\
                DEPOSIT_RESOLUTION: PPTypes.VEC3i,\
                VIS_RESOLUTION: PPTypes.VEC2i,\
                DOMAIN_SIZE_MAX: PPTypes.FLOAT_GPU,\
                DOMAIN_MIN: PPTypes.VEC3f,\
                DOMAIN_MAX: PPTypes.VEC3f,\
                DOMAIN_CENTER: PPTypes.VEC3f,\
                RAY_EPSILON: PPTypes.FLOAT_GPU,\
                deposit_field: ti.template(),\
                trace_field: ti.template(),\
                vis_field: ti.template()):
        n_steps = ti.cast(n_ray_steps_f, PPTypes.INT_GPU)
        aspect_ratio = ti.cast(VIS_RESOLUTION[0], PPTypes.FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], PPTypes.FLOAT_GPU)
        screen_distance = DOMAIN_SIZE_MAX
        camera_offset = camera_distance * PPTypes.VEC3f(ti.cos(camera_azimuth) * ti.sin(camera_polar), ti.sin(camera_azimuth) * ti.sin(camera_polar), ti.cos(camera_polar))
        camera_pos = DOMAIN_CENTER + camera_offset
        cam_Z = timath.normalize(-camera_offset)
        cam_Y = PPTypes.VEC3f(0.0, 0.0, 1.0)
        cam_X = timath.normalize(timath.cross(cam_Z, cam_Y))
        cam_Y = timath.normalize(timath.cross(cam_X, cam_Z))

        for x, y in ti.ndrange(VIS_RESOLUTION[0], VIS_RESOLUTION[1]):
            ## Compute x and y ray directions in neutral camera position
            rx = DOMAIN_SIZE_MAX * (ti.cast(x, PPTypes.FLOAT_GPU) / ti.cast(VIS_RESOLUTION[0], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
            ry = DOMAIN_SIZE_MAX * (ti.cast(y, PPTypes.FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
            ry /= aspect_ratio

            ## Initialize ray origin and direction
            screen_pos = camera_pos + rx * cam_X + ry * cam_Y + screen_distance * cam_Z
            ray_dir = timath.normalize(screen_pos - camera_pos)

            ## Get intersection of the ray with the volume AABB
            t = self.ray_AABB_intersection(camera_pos, ray_dir, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX))
            ray_L = PPTypes.VEC3f(0.0, 0.0, 0.0)
            ray_delta = 1.71 * DOMAIN_SIZE_MAX / n_ray_steps_f

            ## Check if we intersect the volume AABB at all
            if t[1] >= 0.0:
                t[0] += RAY_EPSILON
                t[1] -= RAY_EPSILON
                t_current = t[0] + ti.random(dtype=PPTypes.FLOAT_GPU) * ray_delta
                ray_pos = camera_pos + t_current * ray_dir

                ## Main integration loop
                for i in ti.ndrange(n_steps):
                    if t_current >= t[1]:
                        break
                    trace_val = trace_field[self.world_to_grid_3D(ray_pos, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(TRACE_RESOLUTION))][0]
                    deposit_val = deposit_field[self.world_to_grid_3D(ray_pos, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
                    ray_L += PPTypes.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)) / n_ray_steps_f
                    ray_pos += ray_delta * ray_dir
                    t_current += ray_delta

            vis_field[x, y] = timath.pow(ray_L, 1.0/2.2)
        return
