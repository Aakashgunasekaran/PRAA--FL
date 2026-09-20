import random


def select_clients(num_clients: int, clients_per_round: int, seed: int, round_number: int):
    if not 0 < clients_per_round <= num_clients:
        raise ValueError("clients_per_round must be in (0, num_clients]")
    rng = random.Random(seed + round_number)
    return sorted(rng.sample(range(num_clients), clients_per_round))
