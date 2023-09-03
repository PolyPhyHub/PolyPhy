from polyphy_functions import PolyPhyWindow, PostSimulation, SimulationVisuals, FieldVariables, Agents, TypeAliases, PPVariables, PPData
from kernels import Kernels
from numpy.random import default_rng
import taichi as ti

class PolyPhy:
    def __init__(self):
        ## Initialize Taichi
        ti.init(arch=ti.cpu)
        self.rng = default_rng()
        self.ppData = PPData()
        self.ppVariables = PPVariables(self.ppData)
        self.agents = Agents(self.rng,self.ppVariables,self.ppData)
        self.fieldVariables = FieldVariables(self.ppVariables,self.ppData)
        self.k = Kernels()
        self.simulationVisuals = SimulationVisuals(self.k,self.ppVariables,self.ppData,self.agents,self.fieldVariables)

    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)

PolyPhy().start_simulation()