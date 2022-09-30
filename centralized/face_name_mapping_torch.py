
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import Tensor

import pickle
import os
from tqdm import tqdm
import numpy as np

NUM_PARTITIONS = 2
class Net(nn.Module):
    """Simple Linear layer"""
    def __init__(self, in_dim: int = 128, out_dim: int = 992) -> None:
        super(Net, self).__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        return self.model(x)


def load_data() -> tuple[DataLoader, DataLoader, dict]:
    """Load face embeddings"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "embeddings", "embeddings.pkl")
    with open(pkl_path, "rb") as fp:  
        imgs_ndarray = np.stack(pickle.load(fp))

    # imgs_ndarray = np.random.rand(992, 128)
    img_tensor = torch.tensor(imgs_ndarray, dtype=torch.float32)
    labels = torch.arange(0, img_tensor.size(0))
    trainset = TensorDataset(img_tensor, labels)
    num_examples = {"trainset": len(labels), "testset": len(labels)}
    return (trainset, trainset, num_examples)

def load_partition(idx: int) -> tuple[Dataset, Dataset, dict]:
    """Load 1dx/{NUM_PARTITIONS}th of the training and test data to simulate a partition."""
    assert idx in range(NUM_PARTITIONS)
    trainset, testset, num_examples = load_data()
    n_train = num_examples["trainset"] // NUM_PARTITIONS
    n_test = num_examples["testset"] // NUM_PARTITIONS

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition, num_examples)


def train(
    net: Net,
    trainset: Dataset,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the network."""
    # Create dataloaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (img_batch, labels) in enumerate(trainloader):
            img_batch, labels = img_batch.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def test(net: Net, test_data: DataLoader, DEVICE: torch.device) -> tuple[float, float]:
    """Validate the model on the test set."""
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return (loss / len(testloader.dataset), correct / total)

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainset, testset, num_examples = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net, trainset, 100, DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net, testset, DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)