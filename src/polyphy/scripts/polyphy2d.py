import numpy as np
import taichi as ti
from numpy.random import default_rng

from ..lib.simulator import Poly


class polyphys(Poly):
    """Polyphy2D simulator"""

    name = "polyphy2d"

    def run(self):
        """Simulates 2d run"""
        N_AGENTS = 1000000
        DOMAIN_SCALE = 100.0
        TRACE_RESOLUTION = (1024, 1024)
        DEPOSIT_DOWNSCALING_FACTOR = 4

        DEPOSIT_RESOLUTION = (TRACE_RESOLUTION[0] // DEPOSIT_DOWNSCALING_FACTOR,
                              TRACE_RESOLUTION[1] // DEPOSIT_DOWNSCALING_FACTOR)
        DOMAIN_SIZE = (DOMAIN_SCALE, DOMAIN_SCALE * np.float32(TRACE_RESOLUTION[1]) /
                       np.float32(TRACE_RESOLUTION[0]))
        VIS_RESOLUTION = TRACE_RESOLUTION

        # Type aliases
        FLOAT_CPU = np.float32
        INT_CPU = np.int32
        FLOAT_GPU = ti.f32
        INT_GPU = ti.i32

        VEC2i = ti.types.vector(2, INT_GPU)
        VEC3i = ti.types.vector(2, INT_GPU)
        VEC2f = ti.types.vector(2, FLOAT_GPU)
        VEC3f = ti.types.vector(3, FLOAT_GPU)

        # Initializations
        ti.init(arch=ti.gpu)
        rng = default_rng()

        print('Number of agents:', N_AGENTS)
        print('Trace grid resolution:', TRACE_RESOLUTION)
        print('Deposit grid resolution:', DEPOSIT_RESOLUTION)
        print('Simulation domain size:', DOMAIN_SIZE)

        # Initialize agents
        agents = np.zeros(shape=(N_AGENTS, 4), dtype=FLOAT_CPU)
        agents[:, 0] = rng.uniform(low=0.0, high=DOMAIN_SIZE[0], size=agents.shape[0])
        agents[:, 1] = rng.uniform(low=0.0, high=DOMAIN_SIZE[1], size=agents.shape[0])
        agents[:, 2] = rng.uniform(low=0.0, high=2.0 * np.pi, size=agents.shape[0])
        agents[:, 3] = 1.0

        # TODO load input data for fitting

        # Allocate GPU memory fields
        agents_field = ti.Vector.field(n=4, dtype=FLOAT_GPU, shape=N_AGENTS)
        deposit_field = ti.Vector.field(n=2, dtype=FLOAT_GPU, shape=DEPOSIT_RESOLUTION)
        trace_field = ti.Vector.field(n=1, dtype=FLOAT_GPU, shape=TRACE_RESOLUTION)
        vis_field = ti.Vector.field(n=3, dtype=FLOAT_GPU, shape=VIS_RESOLUTION)

        # Define all GPU kernels and functions
        @ti.kernel
        def zero_field(f: ti.template()):
            for cell in ti.grouped(f):
                f[cell].fill(0.0)
            return

        @ti.kernel
        def copy_field(dst: ti.template(), src: ti.template()):
            for cell in ti.grouped(dst):
                dst[cell] = src[cell]
            return

        @ti.func
        def world_to_grid_2D(pos_world, size_world, size_grid) -> VEC2i:
            return ti.cast((pos_world / size_world) * ti.cast(size_grid, FLOAT_GPU),
                           INT_GPU)

        @ti.func
        def angle_to_dir_2D(angle) -> VEC2f:
            return VEC2f(ti.cos(angle), ti.sin(angle))

        @ti.kernel
        def propagation_step(sense_distance: FLOAT_GPU, sense_angle: FLOAT_GPU,
                             steering_rate: FLOAT_GPU, step_size: FLOAT_GPU,
                             weight_multiplier: FLOAT_GPU):
            for agent in ti.ndrange(agents_field.shape[0]):
                pos = VEC2f(0.0, 0.0)
                pos[0], pos[1], angle, weight = agents_field[agent]

                dir_fwd = angle_to_dir_2D(angle)
                angle_mut = angle + (ti.random(dtype=FLOAT_GPU) - 0.5) * sense_angle
                dir_mut = angle_to_dir_2D(angle_mut)

                # TODO deposit field ping pong
                deposit_fwd = deposit_field[world_to_grid_2D(
                    pos + sense_distance * dir_fwd, VEC2f(DOMAIN_SIZE),
                    VEC2i(DEPOSIT_RESOLUTION))][0]
                deposit_mut = deposit_field[world_to_grid_2D(
                    pos + sense_distance * dir_mut, VEC2f(DOMAIN_SIZE),
                    VEC2i(DEPOSIT_RESOLUTION))][0]

                # TODO domain wrapping
                angle_new = (angle) if (deposit_fwd > deposit_mut) else (
                    steering_rate * angle_mut + (1.0 - steering_rate) * angle)
                dir_new = angle_to_dir_2D(angle_new)
                pos_new = pos + step_size * dir_new

                agents_field[agent][0] = pos_new[0]
                agents_field[agent][1] = pos_new[1]
                agents_field[agent][2] = angle_new

                deposit_cell = world_to_grid_2D(pos_new, VEC2f(DOMAIN_SIZE),
                                                VEC2i(DEPOSIT_RESOLUTION))
                deposit_field[deposit_cell][0] += weight_multiplier * weight

                trace_cell = world_to_grid_2D(pos_new, VEC2f(DOMAIN_SIZE),
                                              VEC2i(TRACE_RESOLUTION))
                trace_field[trace_cell][0] += weight_multiplier * weight
            return

        @ti.kernel
        def relaxation_step_deposit(attenuation: FLOAT_GPU):
            for cell in ti.grouped(deposit_field):
                # TODO deposit diffusion
                deposit_field[cell][0] *= attenuation
            return

        @ti.kernel
        def relaxation_step_trace(attenuation: FLOAT_GPU):
            for cell in ti.grouped(trace_field):
                trace_field[cell][0] *= attenuation
            return

        @ti.kernel
        def render_visualization():
            for x, y in ti.ndrange(vis_field.shape[0], vis_field.shape[1]):
                deposit_val = deposit_field[x * DEPOSIT_RESOLUTION[0] //
                                            VIS_RESOLUTION[0], y *
                                            DEPOSIT_RESOLUTION[1] // VIS_RESOLUTION[1]][0]
                trace_val = trace_field[x * TRACE_RESOLUTION[0] // VIS_RESOLUTION[0],
                                        y * TRACE_RESOLUTION[1] // VIS_RESOLUTION[1]]
                vis_field[x, y] = VEC3f(trace_val, trace_val, deposit_val if
                                        (trace_val < 0.1) else 0.0)
            return

        # Initialize GPU fields
        agents_field.from_numpy(agents)
        zero_field(deposit_field)
        zero_field(trace_field)
        zero_field(vis_field)

        # Main simulation & vis loop
        sense_distance = 1.0
        sense_angle = 2.5
        step_size = 0.1
        attenuation = 0.95
        weight_multiplier = 0.1
        steering_rate = 0.5

        window = ti.ui.Window('PolyPhy', (vis_field.shape[0], vis_field.shape[1]),
                              show_window=True)
        canvas = window.get_canvas()

        while window.running:
            window.GUI.begin('Params', 0.0, 0.0, 0.6, 0.25)
            sense_distance = window.GUI.slider_float('Sense dist', sense_distance, 0.1,
                                                     10.0)
            sense_angle = window.GUI.slider_float('Sense angle', sense_angle, 0.1, 10.0)
            step_size = window.GUI.slider_float('Step size', step_size, 0.01, 0.5)
            attenuation = window.GUI.slider_float('Attenuation', attenuation, 0.9, 0.999)
            weight_multiplier = window.GUI.slider_float('Weight mul', weight_multiplier,
                                                        0.01, 1.0)
            window.GUI.end()

            propagation_step(sense_distance, sense_angle, steering_rate, step_size,
                             weight_multiplier)
            relaxation_step_deposit(attenuation)
            relaxation_step_trace(attenuation)

            render_visualization()
            canvas.set_image(vis_field)
            window.show()

        window.destroy()
        deposit = deposit_field.to_numpy()
        trace = trace_field.to_numpy()

        # TODO store resulting fields
