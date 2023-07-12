import kernels
import taichi as ti
import first
import fourth

@ti.data_oriented
class FinalKernels(kernels.Kernels):
    @ti.kernel
    def render_visualization(self,deposit_vis: first.FLOAT_GPU, trace_vis: first.FLOAT_GPU, current_deposit_index: first.INT_GPU):
        print("Prashant's inherit works!")
        for x, y in ti.ndrange(fourth.vis_field.shape[0], fourth.vis_field.shape[1]):
            deposit_val = fourth.deposit_field[x * fourth.DEPOSIT_RESOLUTION[0] // fourth.VIS_RESOLUTION[0], y * fourth.DEPOSIT_RESOLUTION[1] // fourth.VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = fourth.trace_field[x * fourth.TRACE_RESOLUTION[0] // fourth.VIS_RESOLUTION[0], y * fourth.TRACE_RESOLUTION[1] // fourth.VIS_RESOLUTION[1]]
            fourth.vis_field[x, y] = ti.pow(first.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return