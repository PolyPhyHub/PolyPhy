from second import PolyphyEnums

class SimulationConstants:
    ## Simulation-wide constants
    N_DATA_DEFAULT = 1000
    N_AGENTS_DEFAULT = 1000000
    DOMAIN_SIZE_DEFAULT = (100.0, 100.0)
    TRACE_RESOLUTION_MAX = 1400
    DEPOSIT_DOWNSCALING_FACTOR = 1
    STEERING_RATE = 0.5
    MAX_DEPOSIT = 10.0
    DOMAIN_MARGIN = 0.05

class StateFlags:
    ## State flags
    distance_sampling_distribution = PolyphyEnums.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
    directional_sampling_distribution = PolyphyEnums.EnumDirectionalSamplingDistribution.CONE
    directional_mutation_type = PolyphyEnums.EnumDirectionalMutationType.PROBABILISTIC
    deposit_fetching_strategy = PolyphyEnums.EnumDepositFetchingStrategy.NN_PERTURBED
    agent_boundary_handling = PolyphyEnums.EnumAgentBoundaryHandling.WRAP