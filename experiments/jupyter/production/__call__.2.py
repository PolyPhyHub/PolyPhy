from polyphy import PolyPhy
from polyphy_functions import FieldVariables

class FinalFieldVariables(FieldVariables):
    def __init__(self, dataLoaders, derivedVariables):
        super().__init__(dataLoaders, derivedVariables)
        print("????????????????Something added??????????????????????")

class FinalPolyPhy(PolyPhy):
    def __init__(self):
        super().__init__()
        self.fieldVariables = FinalFieldVariables(self.dataLoaders,self.derivedVariables)

FinalPolyPhy().start_simulation()
