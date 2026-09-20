"""Phase 3 adapter over the canonical Phase 2 partition implementation."""

from federated.client_data import build_partition, get_client_dataset
from federated.flame_config import (
    DIRICHLET_BETA,
    MIN_SAMPLES_PER_CLIENT,
    NUM_CLIENTS,
    RANDOM_SEED,
)


PARTITION = build_partition(
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_BETA,
    seed=RANDOM_SEED,
    min_samples_per_client=MIN_SAMPLES_PER_CLIENT,
)


def get_flame_client_dataset(client_id: int):
    """Return the exact Phase 2-created dataset assigned to a FLAME client."""
    return get_client_dataset(client_id, partition=PARTITION)
