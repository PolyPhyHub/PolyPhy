from main import PolyPhyWindow
from final import SimulationVisuals 
from finalKernels import FinalKernels

class PolyPhy:
    def __init__(self, k = FinalKernels()):
        SimulationVisuals(k)
        PolyPhyWindow(k)