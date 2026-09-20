import traceback
import tests.test_phase2 as t

TESTS = [
    t.test_partition_completeness,
    t.test_no_duplicate_indices,
    t.test_no_missing_indices,
    t.test_indices_valid_range,
    t.test_same_seed_identical_partition,
    t.test_different_seed_different_partition,
    t.test_client_dataset_and_dataloader_creation,
    t.test_client_invalid_id,
    t.test_edge_cases,
]

results = {}
for fn in TESTS:
    name = fn.__name__
    try:
        fn()
        results[name] = ("PASS", None)
    except AssertionError as e:
        results[name] = ("FAIL", str(e))
    except Exception as e:
        results[name] = ("ERROR", traceback.format_exc())

print("Phase 2 Tests")
print("=============")
for name, (status, info) in results.items():
    print(f"{name}: {status}")
    if info:
        print(info)

# Summary
passes = sum(1 for s, _ in results.values() if s == "PASS")
print(f"\nSummary: {passes}/{len(TESTS)} tests passed")
