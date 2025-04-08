import numpy as np
import sympy


def sum_partition_greedy(arr: np.ndarray, n_part: int = 2):
    """Partitions numbers into M subsets using Largest-First Decreasing (LFD)."""
    numbers = sorted(
        [(arr.flat[i], i) for i in range(arr.size)], reverse=True, key=lambda v: v[0]
    )  # Sort largest first
    sums = np.zeros((n_part,), dtype=np.float64)  # Track partition sums
    # print(numbers)

    partitions = np.zeros_like(arr, dtype=np.int64)

    for num, i in numbers:
        idx = np.argmin(sums)  # Find partition with smallest sum
        sums[idx] += num  # Update sum
        partitions.flat[i] = idx
    #     print(f"{i}, {idx}, {sums}")
    # print(partitions)

    return partitions


def int_factor_divide(N: int, n_part: int = 2):
    factors = sympy.factorint(N)
    factor_list = []
    for f, p in factors.items():
        factor_list += [f] * p
    factor_list = np.array(factor_list, dtype=np.int64)
    assert factor_list.prod() == N
    partition = sum_partition_greedy(
        np.log2((factor_list * 2).astype(np.float64)), n_part
    )
    prods = np.array(
        [np.prod(factor_list[partition == i]) for i in range(n_part)], dtype=np.int64
    )
    assert prods.prod() == N
    return prods


if __name__ == "__main__":

    for i in np.arange(2, 201) * 64:
        prods = int_factor_divide(i)
        diffv = prods.max() - prods.min()
        if diffv > np.sqrt(i) * 0:
            print(f"{i:6}: {str(prods):32}: {diffv}")
    print(int_factor_divide(12))
    
    arr = np.array([1, 3, 5, 7, 9, 11])
    arr = np.sort(arr)
    target = 11

    # searchsorted gives the insertion point for target
    index = np.searchsorted(arr, target, side="right") - 1
    print(index)
