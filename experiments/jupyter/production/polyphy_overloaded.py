from polyphy_functions import StateFlags, PolyPhyWindow, PostSimulation, SimulationVisuals, DataLoader, FieldVariables, DerivedVariables, Agents, TypeAliases, SimulationConstants
from kernels import Kernels

class PolyPhy:
    def __init__(self, dataLoaders, derviedVariables, agents, fieldVariables, k, simulationVisuals):
        self.dataLoaders = dataLoaders
        self.derivedVariables = derviedVariables
        self.agents = agents
        self.fieldVariables = fieldVariables
        self.k = k
        self.simulationVisuals = simulationVisuals
        
    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)