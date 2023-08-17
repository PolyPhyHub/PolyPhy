from polyphy_functions import StateFlags, PolyphyEnums, PolyPhyWindow, PostSimulation, SimulationVisuals, DataLoader, FieldVariables, DerivedVariables, Agents, TypeAliases, SimulationConstants
from kernels import Kernels

class PolyPhy:
    def __init__(self, dataLoaders = DataLoader()):

        TypeAliases.change_precision(float_precision="float32", int_precision="int32")

        print()
        print("Before changes:")
        print("N_DATA_DEFAULT:", SimulationConstants.N_DATA_DEFAULT)
        print("N_AGENTS_DEFAULT:", SimulationConstants.N_AGENTS_DEFAULT)
        print("distance_sampling_distribution:", StateFlags.distance_sampling_distribution)
        print("directional_sampling_distribution:", StateFlags.directional_sampling_distribution)

        SimulationConstants.set_constant("N_DATA_DEFAULT", 1500)
        SimulationConstants.set_constant("N_AGENTS_DEFAULT", 2000000)
        StateFlags.set_flag("distance_sampling_distribution", PolyphyEnums.EnumDistanceSamplingDistribution.CONSTANT)
        StateFlags.set_flag("directional_sampling_distribution", PolyphyEnums.EnumDirectionalSamplingDistribution.DISCRETE)

        print("\nAfter changes:")
        print("N_DATA_DEFAULT:", SimulationConstants.N_DATA_DEFAULT)
        print("N_AGENTS_DEFAULT:", SimulationConstants.N_AGENTS_DEFAULT)
        print("distance_sampling_distribution:", StateFlags.distance_sampling_distribution)
        print("directional_sampling_distribution:", StateFlags.directional_sampling_distribution)
        print()

        self.dataLoaders = dataLoaders
        self.derivedVariables = DerivedVariables(self.dataLoaders)
        self.agents = Agents(self.dataLoaders,self.derivedVariables)
        self.fieldVariables = FieldVariables(self.dataLoaders,self.derivedVariables)
        self.k = Kernels(self.derivedVariables,self.fieldVariables, self.dataLoaders)
        self.simulationVisuals = SimulationVisuals(self.k,dataLoaders,self.derivedVariables,self.agents,self.fieldVariables)
        
    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)