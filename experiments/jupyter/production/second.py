from enum import IntEnum

class PolyphyEnums:
    ## Distance sampling distribution for agents
    class EnumDistanceSamplingDistribution(IntEnum):
        CONSTANT = 0
        EXPONENTIAL = 1
        MAXWELL_BOLTZMANN = 2

    ## Directional sampling distribution for agents
    class EnumDirectionalSamplingDistribution(IntEnum):
        DISCRETE = 0
        CONE = 1

    ## Sampling strategy for directional agent mutation
    class EnumDirectionalMutationType(IntEnum):
        DETERMINISTIC = 0
        PROBABILISTIC = 1

    ## Deposit fetching strategy
    class EnumDepositFetchingStrategy(IntEnum):
        NN = 0
        NN_PERTURBED = 1

    ## Handling strategy for agents that leave domain boundary
    class EnumAgentBoundaryHandling(IntEnum):
        WRAP = 0
        REINIT_CENTER = 1
        REINIT_RANDOMLY = 2