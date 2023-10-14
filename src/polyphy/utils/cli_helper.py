import argparse

from core.common import PPConfig
from core.discrete2D import PPConfig_2DDiscrete
from core.discrete3D import PPConfig_3DDiscrete
from pipelines.discrete2D import PolyPhy_2DDiscrete
from pipelines.discrete3D import PolyPhy_3DDiscrete


class CliHelper:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="PolyPhy")
        parser.add_argument(
            "pipeline",
            type=str,
            choices=['2d_discrete', '3d_discrete'],
            help="Run one of the pipelines")
        parser.add_argument(
            '-f',
            '--input-file',
            type=str,
            help="Main input data file (string relative to the root directory)")
        parser.add_argument(
            '-b',
            '--batch-mode',
            action='store_true',
            help="Enable batch mode (run simulations in batch mode)")
        parser.add_argument(
            '-n',
            '--num-iterations',
            type=int,
            help="Number of iterations (specify the number of simulation iterations)")
        parser.add_argument(
            '-d',
            '--sensing-dist',
            type=float,
            help="Sensing distance (set the agent's sensing distance in pixels)")
        parser.add_argument(
            '-a',
            '--sensing-angle',
            type=float,
            help="Sensing angle (set the agent's sensing angle in degrees)")
        parser.add_argument(
            '-e',
            '--sampling-expo',
            type=float,
            help="Sampling exponent (adjust the sampling exponent for data collection)")
        parser.add_argument(
            '-s',
            '--step-size',
            type=float,
            help="Step size (define the step size for agent movement)")
        parser.add_argument(
            '-D',
            '--data-deposit',
            type=str,
            help="Data deposit (specify the method for data deposition)")
        parser.add_argument(
            '-A',
            '--agent-deposit',
            type=str,
            help="Agent deposit (specify the method for agent deposition)")
        parser.add_argument(
            '-X',
            '--deposit-attenuation',
            type=float,
            help="Deposit attenuation (adjust the deposit attenuation factor)")
        parser.add_argument(
            '-T',
            '--trace-attenuation',
            type=float,
            help="Trace attenuation (adjust the trace attenuation factor)")
        parser.add_argument(
            '-Z',
            '--deposit-visualization',
            type=str,
            help="Deposit visualization (select the method for deposit visualization)")
        parser.add_argument(
            '-Y',
            '--trace-visualization',
            type=str,
            help="Trace visualization (select the method for trace visualization)")
        parser.add_argument(
            '--distance-distribution',
            type=str,
            choices=['constant', 'exponential', 'maxwell-boltzmann'],
            help="Distance distribution \
                (choose the type of distance distribution for agent interactions)")
        parser.add_argument(
            '--directional-distribution',
            type=str,
            choices=['discrete', 'cone'],
            help="Directional distribution \
                (choose the type of directional distribution for agent movement)")
        parser.add_argument(
            '--directional-mutation',
            type=str,
            choices=['deterministic', 'stochastic'],
            help="Directional mutation \
                (choose the type of directional mutation for agent behavior)")
        parser.add_argument(
            '--deposit-fetching',
            type=str,
            choices=['nearest-neighbor', 'noise-perturbed-NN'],
            help="Deposit fetching (select the method for deposit fetching)")
        parser.add_argument(
            '--agent-boundary-handling',
            type=str,
            choices=['wrap-around', 're-initialize-center', 're-initialize-randomly'],
            help="Agent boundary handling \
                (specify how agents interact with boundaries)")
        CliHelper.args = parser.parse_args()

    @staticmethod
    def parse_values(ppConfig):
        args = CliHelper.args
        if args.pipeline == "2d_discrete":
            ppConfig = PPConfig_2DDiscrete()
        elif args.pipeline == "3d_discrete":
            ppConfig = PPConfig_3DDiscrete()
        if args.input_file:
            ppConfig.setter("input_file", str(args.input_file))
        else:
            ppConfig.setter("input_file", '')
            raise AssertionError("Please specify the main input data file \
                                    (string relative to the root directory)")
        if args.batch_mode:
            print("Batch mode activated!")
            if args.num_iterations:
                print(f"Number of iterations: {int(args.num_iterations)}")
            else:
                raise AssertionError("Please set number of iterations for batch mode \
                                     using -n <int>")
        if args.num_iterations and not args.batch_mode:
            raise AssertionError("Please set to batch mode")
        if args.sensing_dist:
            ppConfig.setter("sense_distance", args.sensing_dist)
        if args.sensing_angle:
            ppConfig.setter("sense_angle", args.sensing_angle)
        if args.sampling_expo:
            ppConfig.setter("sampling_exponent", args.sampling_expo)
        if args.step_size:
            ppConfig.setter("step_size", args.step_size)
        if args.data_deposit:
            ppConfig.setter("data_deposit", args.data_deposit)
        if args.agent_deposit:
            ppConfig.setter("agent_deposit", args.agent_deposit)
        if args.deposit_attenuation:
            ppConfig.setter("deposit_attenuation", args.deposit_attenuation)
        if args.trace_attenuation:
            ppConfig.setter("trace_attenuation", args.trace_attenuation)
        if args.deposit_visualization:
            ppConfig.setter("deposit_vis", args.deposit_visualization)
        if args.trace_visualization:
            ppConfig.setter("trace_vis", args.trace_visualization)
        if args.distance_distribution:
            if args.distance_distribution == "constant":
                ppConfig.setter("distance_sampling_distribution",
                                PPConfig.EnumDistanceSamplingDistribution.CONSTANT)
            elif args.distance_distribution == "exponential":
                ppConfig.setter("distance_sampling_distribution",
                                PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL)
            elif args.distance_distribution == "maxwell-boltzmann":
                ppConfig.setter("distance_sampling_distribution",
                                PPConfig.EnumDistanceSamplingDistribution.
                                MAXWELL_BOLTZMANN)
        if args.directional_distribution:
            if args.directional_distribution == "discrete":
                ppConfig.setter("directional_sampling_distribution",
                                PPConfig.EnumDirectionalSamplingDistribution.DISCRETE)
            elif args.directional_distribution == "cone":
                ppConfig.setter("directional_sampling_distribution",
                                PPConfig.EnumDirectionalSamplingDistribution.CONE)
        if args.directional_mutation:
            if args.directional_mutation == "deterministic":
                ppConfig.setter("directional_mutation_type",
                                PPConfig.EnumDirectionalMutationType.DETERMINISTIC)
            elif args.directional_mutation == "stochastic":
                ppConfig.setter("directional_mutation_type",
                                PPConfig.EnumDirectionalMutationType.PROBABILISTIC)
        if args.deposit_fetching:
            if args.deposit_fetching == "nearest-neighbor":
                ppConfig.setter("deposit_fetching_strategy",
                                PPConfig.EnumDepositFetchingStrategy.NN)
            elif args.deposit_fetching == "noise-perturbed-NN":
                ppConfig.setter("deposit_fetching_strategy",
                                PPConfig.EnumDepositFetchingStrategy.NN_PERTURBED)
        if args.agent_boundary_handling:
            if args.agent_boundary_handling == "wrap-around":
                ppConfig.setter("agent_boundary_handling",
                                PPConfig.EnumAgentBoundaryHandling.WRAP)
            elif args.agent_boundary_handling == "re-initialize-center":
                ppConfig.setter("agent_boundary_handling",
                                PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER)
            elif args.agent_boundary_handling == "re-initialize-randomly":
                ppConfig.setter("agent_boundary_handling",
                                PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY)
        if args.pipeline == "2d_discrete":
            PolyPhy_2DDiscrete(ppConfig).start_simulation()
        elif args.pipeline == "3d_discrete":
            PolyPhy_3DDiscrete(ppConfig).start_simulation()
