from torch.utils.data import Dataset
import torch

from centralized.face_name_mapping_torch import load_data


def load_partition(
    idx: int,
    num_partitions: int
) -> tuple[Dataset, Dataset, dict]:
    """Load {idx}/{num_partitions}th of the training and test data to simulate a partition."""
    assert idx in range(num_partitions)
    trainset, testset, num_examples = load_data()
    n_train = num_examples["trainset"] // num_partitions
    n_test = num_examples["testset"] // num_partitions

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition, num_examples)