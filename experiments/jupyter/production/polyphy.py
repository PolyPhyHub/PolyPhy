from main import PolyPhyWindow, PostSimulation
from final import SimulationVisuals 
from kernels import Kernels
from fourth import DataLoader, FieldVariables, DerivedVariables, Agents

class PolyPhy:
    def __init__(self, dataLoaders = DataLoader()):
        self.dataLoaders = dataLoaders
        self.derivedVariables = DerivedVariables(self.dataLoaders)
        self.agents = Agents(self.dataLoaders,self.derivedVariables)
        self.fieldVariables = FieldVariables(self.dataLoaders,self.derivedVariables)
        self.k = Kernels(self.derivedVariables,self.fieldVariables, self.dataLoaders)
        self.simulationVisuals = SimulationVisuals(self.k,dataLoaders,self.derivedVariables,self.agents,self.fieldVariables)
        
    def start_simulation(self):
        PolyPhyWindow(self.k,self.simulationVisuals)
        PostSimulation(self.simulationVisuals)