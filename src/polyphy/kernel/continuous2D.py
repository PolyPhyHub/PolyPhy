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
