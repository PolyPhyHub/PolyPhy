# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath
from core.common import PPTypes


@ti.data_oriented
class PPKernels:

    # GPU functions (callable by kernels) ===========================================
    @ti.func
    def custom_mod(self, a, b) -> PPTypes.FLOAT_GPU:
        return a - b * ti.floor(a / b)
    
    @ti.func
    def angle_to_dir_2D(self, angle) -> PPTypes.VEC2f:
        return timath.normalize(PPTypes.VEC2f(ti.cos(angle), ti.sin(angle)))
    
    @ti.func
    def world_to_grid_2D(
            self,
            pos_world,
            domain_min,
            domain_max,
            grid_resolution) -> PPTypes.VEC2i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(
            grid_resolution, PPTypes.FLOAT_GPU), PPTypes.INT_GPU)
        return ti.max(PPTypes.VEC2i(0, 0), ti.min(grid_coord, grid_resolution - (1, 1)))

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

    # GPU kernels (callable by core classes via Taichi API) ========================
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
    def deposit_relaxation_step_2D(
            self,
            attenuation: PPTypes.FLOAT_GPU,
            current_deposit_index: PPTypes.INT_GPU,
            DEPOSIT_RESOLUTION: PPTypes.VEC2i,
            deposit_field: ti.template()):
        DIFFUSION_WEIGHTS = [1.0, 1.0, 0.707]
        DIFFUSION_WEIGHTS_NORM = (DIFFUSION_WEIGHTS[0] + 4.0 * DIFFUSION_WEIGHTS[1] + 4.0 * DIFFUSION_WEIGHTS[2])
        for cell in ti.grouped(deposit_field):
            # The "beautiful" expression below implements
            # a 3x3 kernel diffusion with manually wrapped addressing
            # Taichi doesn't support modulo for tuples
            # so each dimension is handled separately
            value = DIFFUSION_WEIGHTS[0] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]
            deposit_field[cell][1 - current_deposit_index] = (attenuation * value / DIFFUSION_WEIGHTS_NORM)
        return

    @ti.kernel
    def trace_relaxation_step_2D(
            self,
            attenuation: PPTypes.FLOAT_GPU,
            trace_field: ti.template()):
        for cell in ti.grouped(trace_field):
            # Perturb the attenuation by a small factor
            # to avoid accumulating quantization errors
            trace_field[cell][0] *= (attenuation - 0.001 + 0.002 * ti.random(dtype=PPTypes.FLOAT_GPU))
        return
    
    @ti.kernel
    def render_visualization_2D(
                self,
                trace_vis: PPTypes.FLOAT_GPU,
                deposit_vis: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                TRACE_RESOLUTION: PPTypes.VEC2i,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                VIS_RESOLUTION: PPTypes.VEC2i,
                trace_field: ti.template(),
                deposit_field: ti.template(),
                vis_field: ti.template()):
        for x, y in ti.ndrange(vis_field.shape[0], vis_field.shape[1]):
            deposit_val = deposit_field[
                x * DEPOSIT_RESOLUTION[0] // VIS_RESOLUTION[0],
                y * DEPOSIT_RESOLUTION[1] // VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = trace_field[
                x * TRACE_RESOLUTION[0] // VIS_RESOLUTION[0],
                y * TRACE_RESOLUTION[1] // VIS_RESOLUTION[1]]
            vis_field[x, y] = ti.pow(
                PPTypes.VEC3f(
                    trace_vis * trace_val,
                    deposit_vis * deposit_val,
                    ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return
