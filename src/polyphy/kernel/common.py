import taichi as ti

from core.common import PPTypes


@ti.data_oriented
class PPKernels:

    # GPU functions (callable by kernels) ===========================================
    @ti.func
    def custom_mod(self, a, b) -> PPTypes.FLOAT_GPU:
        return a - b * ti.floor(a / b)

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
        return PPTypes.VEC2f(-1.0, -1.0) if (
            t7 < 0.0 or t6 >= t7) else PPTypes.VEC2f(t6, t7)

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
