import kernels
import taichi as ti
from first import TypeAliases
from fourth import FieldVariables, DerivedVariables, DataLoader

@ti.data_oriented
class FinalKernels(kernels.Kernels):
    @ti.kernel
    def render_visualization(self,deposit_vis: TypeAliases.FLOAT_GPU, trace_vis: TypeAliases.FLOAT_GPU, current_deposit_index: TypeAliases.INT_GPU):
        print("Prashant's inherit works!")
        for x, y in ti.ndrange(FieldVariables.vis_field.shape[0], FieldVariables.vis_field.shape[1]):
            deposit_val = FieldVariables.deposit_field[x * DerivedVariables.DEPOSIT_RESOLUTION[0] // DerivedVariables.VIS_RESOLUTION[0], y * DerivedVariables.DEPOSIT_RESOLUTION[1] // DerivedVariables.VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = FieldVariables.trace_field[x * DerivedVariables.TRACE_RESOLUTION[0] // DerivedVariables.VIS_RESOLUTION[0], y * DerivedVariables.TRACE_RESOLUTION[1] // DerivedVariables.VIS_RESOLUTION[1]]
            FieldVariables.vis_field[x, y] = ti.pow(TypeAliases.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return