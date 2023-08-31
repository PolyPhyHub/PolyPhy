from polyphy_functions import StateFlags, PolyPhyWindow, PostSimulation, SimulationVisuals, DataLoader, FieldVariables, DerivedVariables, Agents, TypeAliases, SimulationConstants
from kernels import Kernels
from numpy.random import default_rng
import taichi as ti

class PolyPhy:
    def __init__(self):
        ## Initialize Taichi
        ti.init(arch=ti.cpu)
        self.rng = default_rng()
        self.dataLoaders = DataLoader(self.rng)
        self.derivedVariables = DerivedVariables(self.dataLoaders)
        self.agents = Agents(self.rng,self.dataLoaders,self.derivedVariables)
        self.fieldVariables = FieldVariables(self.dataLoaders,self.derivedVariables)
        self.k = Kernels()
        self.simulationVisuals = SimulationVisuals(self.k,self.dataLoaders,self.derivedVariables,self.agents,self.fieldVariables)

    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)

PolyPhy().start_simulation()