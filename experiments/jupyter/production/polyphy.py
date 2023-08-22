from polyphy_functions import StateFlags, PolyPhyWindow, PostSimulation, SimulationVisuals, DataLoader, FieldVariables, DerivedVariables, Agents, TypeAliases, SimulationConstants
from kernels import Kernels

class PolyPhy:
    def __init__(self):
        self.dataLoaders = DataLoader()
        self.derivedVariables = DerivedVariables(self.dataLoaders)
        self.agents = Agents(self.dataLoaders,self.derivedVariables)
        self.fieldVariables = FieldVariables(self.dataLoaders,self.derivedVariables)
        self.k = Kernels(self.derivedVariables,self.fieldVariables, self.dataLoaders)
        self.simulationVisuals = SimulationVisuals(self.k,self.dataLoaders,self.derivedVariables,self.agents,self.fieldVariables)

    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)

PolyPhy().start_simulation()