from polyphy_core import *
from kernels import Kernels
import argparse
from numpy.random import default_rng
import taichi as ti

class PolyPhy:

    def start_simulation(self):
        ## specific implementation has to create the following classes defined for that pipeline
        ## PPSimulation(self.ppInternalData,self.batch_mode,self.num_iterations)
        ## PPPostSimulation(self.ppInternalData)
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PolyPhy")
        parser.add_argument('-f', '--input-file', type=str, help="Main input data file (string relative to the root directory)")
        parser.add_argument('-b', '--batch-mode', action='store_true', help="Enable batch mode (run simulations in batch mode)")
        parser.add_argument('-n', '--num-iterations', type=int, help="Number of iterations (specify the number of simulation iterations)")
        parser.add_argument('-d', '--sensing-dist', type=float, help="Sensing distance (set the agent's sensing distance in pixels)")
        parser.add_argument('-a', '--sensing-angle', type=float, help="Sensing angle (set the agent's sensing angle in degrees)")
        parser.add_argument('-e', '--sampling-expo', type=float, help="Sampling exponent (adjust the sampling exponent for data collection)")
        parser.add_argument('-s', '--step-size', type=float, help="Step size (define the step size for agent movement)")
        parser.add_argument('-D', '--data-deposit', type=str, help="Data deposit (specify the method for data deposition)")
        parser.add_argument('-A', '--agent-deposit', type=str, help="Agent deposit (specify the method for agent deposition)")
        parser.add_argument('-X', '--deposit-attenuation', type=float, help="Deposit attenuation (adjust the deposit attenuation factor)")
        parser.add_argument('-T', '--trace-attenuation', type=float, help="Trace attenuation (adjust the trace attenuation factor)")
        parser.add_argument('-Z', '--deposit-visualization', type=str, help="Deposit visualization (select the method for deposit visualization)")
        parser.add_argument('-Y', '--trace-visualization', type=str, help="Trace visualization (select the method for trace visualization)")
        parser.add_argument('--distance-distribution', type=str, choices=['constant', 'exponential', 'maxwell-boltzmann'],
                            help="Distance distribution (choose the type of distance distribution for agent interactions)")
        parser.add_argument('--directional-distribution', type=str, choices=['discrete', 'cone'],
                            help="Directional distribution (choose the type of directional distribution for agent movement)")
        parser.add_argument('--directional-mutation', type=str, choices=['deterministic', 'stochastic'],
                            help="Directional mutation (choose the type of directional mutation for agent behavior)")
        parser.add_argument('--deposit-fetching', type=str, choices=['nearest-neighbor', 'noise-perturbed-NN'],
                            help="Deposit fetching (select the method for deposit fetching)")
        parser.add_argument('--agent-boundary-handling', type=str, choices=['wrap-around', 're-initialize-center', 're-initialize-randomly'],
                            help="Agent boundary handling (specify how agents interact with boundaries)")
        self.args = parser.parse_args()
    
    def parse_values(self):
        if self.args.input_file:
            self.ppConfig.setter("input_file",self.args.input_file)
            if self.args.input_file:
                self.input_file = str(self.args.input_file)
            else:
                self.input_file = ''
                raise AssertionError("Please specify the main input data file (string relative to the root directory)")
        if self.args.batch_mode:
            self.batch_mode = True
            print("Batch mode activated!")
            if self.args.num_iterations:
                self.num_iterations = int(self.args.num_iterations)
                print(f"Number of iterations: {self.args.num_iterations}")
            else:
                raise AssertionError("Please set number of iterations for batch mode")
        if self.args.num_iterations and not self.args.batch_mode:
            raise AssertionError("Please set to batch mode")
        if self.args.sensing_dist:
            self.ppConfig.setter("sense_distance",self.args.sensing_dist)
        if self.args.sensing_angle:
            self.ppConfig.setter("sense_angle",self.args.sensing_angle)
        if self.args.sampling_expo:
            self.ppConfig.setter("sampling_exponent",self.args.sampling_expo)
        if self.args.step_size:
            self.ppConfig.setter("step_size",self.args.step_size)
        if self.args.data_deposit:
            self.ppConfig.setter("data_deposit",self.args.data_deposit)
        if self.args.agent_deposit:
            self.ppConfig.setter("agent_deposit",self.args.agent_deposit)
        if self.args.deposit_attenuation:
            self.ppConfig.setter("deposit_attenuation",self.args.deposit_attenuation)
        if self.args.trace_attenuation:
            self.ppConfig.setter("trace_attenuation",self.args.trace_attenuation)
        if self.args.deposit_visualization:
            self.ppConfig.setter("deposit_vis",self.args.deposit_visualization)
        if self.args.trace_visualization:
            self.ppConfig.setter("trace_vis",self.args.trace_visualization)
        if self.args.distance_distribution:
            if self.args.distance_distribution == "constant":
                self.ppConfig.setter("distance_sampling_distribution",PPConfig.EnumDistanceSamplingDistribution.CONSTANT)
            elif self.args.distance_distribution == "exponential":
                self.ppConfig.setter("distance_sampling_distribution",PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL)
            elif self.args.distance_distribution == "maxwell-boltzmann":
                self.ppConfig.setter("distance_sampling_distribution",PPConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN)           
        if self.args.directional_distribution:
            if self.args.directional_distribution == "discrete":
                self.ppConfig.setter("directional_sampling_distribution",PPConfig.EnumDirectionalSamplingDistribution.DISCRETE)
            elif self.args.directional_distribution == "cone":
                self.ppConfig.setter("directional_sampling_distribution",PPConfig.EnumDirectionalSamplingDistribution.CONE)
        if self.args.directional_mutation:
            if self.args.directional_mutation == "deterministic":
                self.ppConfig.setter("directional_mutation_type",PPConfig.EnumDirectionalMutationType.DETERMINISTIC)
            elif self.args.directional_mutation == "stochastic":
                self.ppConfig.setter("directional_mutation_type",PPConfig.EnumDirectionalMutationType.PROBABILISTIC)
        if self.args.deposit_fetching:
            if self.args.deposit_fetching == "nearest-neighbor":
                self.ppConfig.setter("deposit_fetching_strategy",PPConfig.EnumDepositFetchingStrategy.NN)
            elif self.args.deposit_fetching == "noise-perturbed-NN":
                self.ppConfig.setter("deposit_fetching_strategy",PPConfig.EnumDepositFetchingStrategy.NN_PERTURBED)
        if self.args.agent_boundary_handling:
            if self.args.agent_boundary_handling == "wrap-around":
                self.ppConfig.setter("agent_boundary_handling",PPConfig.EnumAgentBoundaryHandling.WRAP)
            elif self.args.agent_boundary_handling == "re-initialize-center":
                self.ppConfig.setter("agent_boundary_handling",PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER)
            elif self.args.agent_boundary_handling == "re-initialize-randomly":
                self.ppConfig.setter("agent_boundary_handling",PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY)
                