import numpy as np
from collections import defaultdict


def _extract_labels(dataset):
    """Try several common dataset attributes to extract labels without loading images.
    Falls back to iterating the dataset if necessary.
    Returns a 1D numpy array of integer labels.
    """
    # Common attribute names
    for attr in ("labels", "targets", "y", "gt_labels"):
        if hasattr(dataset, attr):
            labels = getattr(dataset, attr)
            try:
                return np.array(labels).reshape(-1)
            except Exception:
                return np.array(labels)

    # Fallback: iterate (may be slower but robust)
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        # item can be (img, label) or dict-like
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lbl = item[1]
        elif isinstance(item, dict) and "label" in item:
            lbl = item["label"]
        else:
            raise RuntimeError("Unable to extract labels from dataset elements")
        # Ensure scalar
        if isinstance(lbl, (list, tuple, np.ndarray)):
            lbl = int(np.array(lbl).reshape(-1)[0])
        else:
            lbl = int(lbl)
        labels.append(lbl)
    return np.array(labels)


def partition_dataset(dataset, num_clients, alpha, seed=42, min_samples_per_client=0):
    """Partition the TRAINING dataset indices into num_clients non-IID groups
    using a Dirichlet distribution with concentration parameter alpha.

    Args:
        dataset: a torch.utils.data.Dataset for the training split
        num_clients: int > 0
        alpha: float > 0 (Dirichlet concentration)
        seed: integer seed for determinism

    Returns:
        dict: {client_id (int): [indices]}
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if min_samples_per_client < 0:
        raise ValueError("min_samples_per_client must be >= 0")

    labels = _extract_labels(dataset)
    n = len(labels)

    rng = np.random.RandomState(seed)

    classes = np.unique(labels)
    num_classes = len(classes)

    # Initialize empty lists for each client
    client_indices = {i: [] for i in range(num_clients)}

    # For each class, split its indices across clients according to a Dirichlet draw
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        # Shuffle class indices deterministically
        rng.shuffle(idx_c)

        # Draw Dirichlet proportions
        proportions = rng.dirichlet([alpha] * num_clients)

        # Convert proportions to integer counts that sum to len(idx_c)
        counts = (proportions * len(idx_c)).astype(int)
        # Fix rounding issues: distribute remaining samples
        diff = len(idx_c) - counts.sum()
        while diff > 0:
            # add 1 to the client with largest proportion until diff==0
            idx = int(np.argmax(proportions))
            counts[idx] += 1
            proportions[idx] = 0  # avoid repeatedly picking same if tie
            diff -= 1
        while diff < 0:
            # remove from client with non-zero count
            idx = int(np.argmax(counts))
            if counts[idx] > 0:
                counts[idx] -= 1
                diff += 1
            else:
                break

        # Now slice idx_c according to counts
        start = 0
        for client_id in range(num_clients):
            cnt = counts[client_id]
            if cnt > 0:
                part = idx_c[start:start + cnt].tolist()
                client_indices[client_id].extend(part)
                start += cnt

    # Final checks: ensure indices are within range and unique
    assigned = np.concatenate([np.array(v, dtype=int) if len(v) > 0 else np.array([], dtype=int) for v in client_indices.values()])
    if assigned.size != 0:
        unique_assigned = np.unique(assigned)
    else:
        unique_assigned = np.array([], dtype=int)

    # Note: some indices may remain unassigned due to rounding; assign leftover indices deterministically
    all_assigned_set = set(unique_assigned.tolist())
    missing = [i for i in range(n) if i not in all_assigned_set]
    if missing:
        # distribute missing one-by-one to clients with smallest current size
        for idx in missing:
            # choose client with smallest assigned count
            smallest = min(client_indices.keys(), key=lambda k: len(client_indices[k]))
            client_indices[smallest].append(int(idx))

    if min_samples_per_client:
        if n < num_clients * min_samples_per_client:
            raise ValueError("dataset is too small to satisfy minimum client sizes")
        for client_id in range(num_clients):
            while len(client_indices[client_id]) < min_samples_per_client:
                donor = max(
                    (candidate for candidate in client_indices if len(client_indices[candidate]) > min_samples_per_client),
                    key=lambda candidate: len(client_indices[candidate]),
                    default=None,
                )
                if donor is None:
                    raise ValueError("Unable to satisfy minimum client partition size")
                client_indices[client_id].append(client_indices[donor].pop())
    return client_indices


def verify_partition(partition, total_len):
    """Verify partition integrity.

    Returns a dict with checks and boolean results.
    """
    results = {}
    # Sum counts
    counts = {k: len(v) for k, v in partition.items()}
    results["sum_counts"] = sum(counts.values()) == total_len

    # Unique assigned indices
    all_indices = []
    for v in partition.values():
        all_indices.extend(v)
    all_indices = np.array(all_indices, dtype=int)
    results["unique_assigned_count"] = len(np.unique(all_indices)) == total_len

    # No missing indices
    results["no_missing"] = set(all_indices.tolist()) == set(range(total_len))

    # Indices in range
    results["indices_in_range"] = all((0 <= idx < total_len) for idx in all_indices)

    # No duplicates across clients
    results["no_duplicates"] = len(all_indices) == len(np.unique(all_indices))

    return results
