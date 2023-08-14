from polyphy import PolyPhy
from finalKernels import FinalKernels

class FinalPolyPhy(PolyPhy):
    def __init__(self):
        super().__init__()
        self.k = FinalKernels(self.derivedVariables,self.fieldVariables,self.dataLoaders)

FinalPolyPhy().start_simulation()
