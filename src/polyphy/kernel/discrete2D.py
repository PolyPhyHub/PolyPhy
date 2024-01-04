# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath

from core.common import PPTypes, PPConfig
from .common import PPKernels


@ti.data_oriented
class PPKernels_2DDiscrete(PPKernels):

    @ti.kernel
    def data_step_2D_discrete(
                self,
                data_deposit: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                DOMAIN_MIN: PPTypes.VEC2f,
                DOMAIN_MAX: PPTypes.VEC2f,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                data_field: ti.template(),
                deposit_field: ti.template()):
        for point in ti.ndrange(data_field.shape[0]):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos[0], pos[1], weight = data_field[point]
            deposit_cell = self.world_to_grid_2D(
                pos,
                PPTypes.VEC2f(DOMAIN_MIN),
                PPTypes.VEC2f(DOMAIN_MAX),
                PPTypes.VEC2i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return
