import kernels
import taichi as ti
from first import TypeAliases

@ti.data_oriented
class FinalKernels(kernels.Kernels):
    @ti.kernel
    def render_visualization(self,deposit_vis: TypeAliases.FLOAT_GPU, trace_vis: TypeAliases.FLOAT_GPU, current_deposit_index: TypeAliases.INT_GPU):
        print("Prashant's inherit works!")
        for x, y in ti.ndrange(self.fieldVariables.vis_field.shape[0], self.fieldVariables.vis_field.shape[1]):
            deposit_val = self.fieldVariables.deposit_field[x * self.derivedVariables.DEPOSIT_RESOLUTION[0] // self.derivedVariables.VIS_RESOLUTION[0], y * self.derivedVariables.DEPOSIT_RESOLUTION[1] // self.derivedVariables.VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = self.fieldVariables.trace_field[x * self.derivedVariables.TRACE_RESOLUTION[0] // self.derivedVariables.VIS_RESOLUTION[0], y * self.derivedVariables.TRACE_RESOLUTION[1] // self.derivedVariables.VIS_RESOLUTION[1]]
            self.fieldVariables.vis_field[x, y] = ti.pow(TypeAliases.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return
    