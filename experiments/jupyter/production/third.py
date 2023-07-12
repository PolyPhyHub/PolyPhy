import second

## Simulation-wide constants
N_DATA_DEFAULT = 1000
N_AGENTS_DEFAULT = 1000000
DOMAIN_SIZE_DEFAULT = (100.0, 100.0)
TRACE_RESOLUTION_MAX = 1400
DEPOSIT_DOWNSCALING_FACTOR = 1
STEERING_RATE = 0.5
MAX_DEPOSIT = 10.0
DOMAIN_MARGIN = 0.05

## State flags
distance_sampling_distribution = second.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
directional_sampling_distribution = second.EnumDirectionalSamplingDistribution.CONE
directional_mutation_type = second.EnumDirectionalMutationType.PROBABILISTIC
deposit_fetching_strategy = second.EnumDepositFetchingStrategy.NN_PERTURBED
agent_boundary_handling = second.EnumAgentBoundaryHandling.WRAP