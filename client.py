"""Flower client example using PyTorch for facial recognition."""
import argparse

import flwr as fl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from centralized import face_name_mapping_torch

USE_FEDBN: bool = True

# Flower Client
class Client(fl.client.NumPyClient):
    """Flower client implementing facial recognition using PyTorch."""
    def __init__(
        self,
        model: face_name_mapping_torch.Net,
        trainset: Dataset,
        testset: Dataset,
        num_examples: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.num_examples = num_examples
        self.device = device

    def get_parameters(self, config: dict[str, str] = None) -> list[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, str]
    ) -> tuple[list[np.ndarray], int, dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)

        face_name_mapping_torch.train(self.model, self.trainset, epochs=30, device=self.device)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, str]
    ) -> tuple[float, int, dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = face_name_mapping_torch.test(self.model, self.testset, self.device)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


if __name__ == "__main__":
    """Load data, start Client."""
    # Load arguments
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 2),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )

    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    # Load data
    trainset, testset, num_examples = face_name_mapping_torch.load_partition(args.partition)


    # Load model
    model = face_name_mapping_torch.Net().to(device).train()

    # # Perform a single forward pass to properly initialize BatchNorm
    # _ = model(next(iter(trainloader))[0].to(device))

    # Start client
    client = Client(model, trainset, testset, num_examples, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
