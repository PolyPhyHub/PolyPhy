from polyphy_functions import PolyPhyWindow, PostSimulation, SimulationVisuals, FieldVariables, Agents, TypeAliases, PPVariables, PPData
from kernels import Kernels
import argparse
from numpy.random import default_rng
import taichi as ti

class PolyPhy:
    def __init__(self):
        self.parse_args()
        ti.init(arch=ti.cpu)
        self.rng = default_rng()
        self.ppData = PPData()
        self.ppVariables = PPVariables(self.ppData)
        self.batch_mode = False
        self.num_iterations = -1
        self.parse_values()
        self.agents = Agents(self.rng,self.ppVariables,self.ppData)
        self.fieldVariables = FieldVariables(self.ppVariables,self.ppData)
        self.k = Kernels()
        self.simulationVisuals = SimulationVisuals(self.k,self.ppVariables,self.ppData,self.agents,self.fieldVariables)

    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals,self.batch_mode,self.num_iterations)
        PostSimulation(self.simulationVisuals)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PolyPhy")
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

        # Parse the command-line arguments
        self.args = parser.parse_args()
    
    def parse_values(self):
        if self.args.batch_mode:
            self.batch_mode = True
            print("Batch mode enabled!!!")
            if self.args.num_iterations:
                self.num_iterations = int(self.args.num_iterations)
                print(f"Number of iterations: {self.args.num_iterations}")
            else:
                raise AssertionError("Please set number of iterations for batch mode")
        if self.args.num_iterations and not self.args.batch_mode:
            raise AssertionError("Please set to batch mode")
        if self.args.sensing_dist:
            self.ppVariables.setter("sense_distance",self.args.sensing_dist)
        if self.args.sensing_angle:
            self.ppVariables.setter("sense_angle",self.args.sensing_angle)
        if self.args.sampling_expo:
            self.ppVariables.setter("sampling_exponent",self.args.sampling_expo)
        if self.args.step_size:
            self.ppVariables.setter("step_size",self.args.step_size)
        if self.args.data_deposit:
            self.ppVariables.setter("data_deposit",self.args.data_deposit)
        if self.args.agent_deposit:
            self.ppVariables.setter("agent_deposit",self.args.agent_deposit)
        if self.args.deposit_attenuation:
            self.ppVariables.setter("deposit_attenuation",self.args.deposit_attenuation)
        if self.args.trace_attenuation:
            self.ppVariables.setter("trace_attenuation",self.args.trace_attenuation)
        if self.args.deposit_visualization:
            self.ppVariables.setter("deposit_vis",self.args.deposit_visualization)
        if self.args.trace_visualization:
            self.ppVariables.setter("trace_vis",self.args.trace_visualization)
        if self.args.distance_distribution:
            if self.args.distance_distribution == "constant":
                self.ppVariables.setter("distance_sampling_distribution",PPVariables.EnumDistanceSamplingDistribution.CONSTANT)
            elif self.args.distance_distribution == "exponential":
                self.ppVariables.setter("distance_sampling_distribution",PPVariables.EnumDistanceSamplingDistribution.EXPONENTIAL)
            elif self.args.distance_distribution == "maxwell-boltzmann":
                self.ppVariables.setter("distance_sampling_distribution",PPVariables.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN)           
        if self.args.directional_distribution:
            if self.args.directional_distribution == "discrete":
                self.ppVariables.setter("directional_sampling_distribution",PPVariables.EnumDirectionalSamplingDistribution.DISCRETE)
            elif self.args.directional_distribution == "cone":
                self.ppVariables.setter("directional_sampling_distribution",PPVariables.EnumDirectionalSamplingDistribution.CONE)
        if self.args.directional_mutation:
            if self.args.directional_mutation == "deterministic":
                self.ppVariables.setter("directional_mutation_type",PPVariables.EnumDirectionalMutationType.DETERMINISTIC)
            elif self.args.directional_mutation == "stochastic":
                self.ppVariables.setter("directional_mutation_type",PPVariables.EnumDirectionalMutationType.PROBABILISTIC)
        if self.args.deposit_fetching:
            if self.args.deposit_fetching == "nearest-neighbor":
                self.ppVariables.setter("deposit_fetching_strategy",PPVariables.EnumDepositFetchingStrategy.NN)
            elif self.args.deposit_fetching == "noise-perturbed-NN":
                self.ppVariables.setter("deposit_fetching_strategy",PPVariables.EnumDepositFetchingStrategy.NN_PERTURBED)
        if self.args.agent_boundary_handling:
            if self.args.agent_boundary_handling == "wrap-around":
                self.ppVariables.setter("agent_boundary_handling",PPVariables.EnumAgentBoundaryHandling.WRAP)
            elif self.args.agent_boundary_handling == "re-initialize-center":
                self.ppVariables.setter("agent_boundary_handling",PPVariables.EnumAgentBoundaryHandling.REINIT_CENTER)
            elif self.args.agent_boundary_handling == "re-initialize-randomly":
                self.ppVariables.setter("agent_boundary_handling",PPVariables.EnumAgentBoundaryHandling.REINIT_RANDOMLY)

PolyPhy().start_simulation()