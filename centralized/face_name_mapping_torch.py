
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor

import pickle
import os
from tqdm import tqdm
import numpy as np

class Net(nn.Module):
    """Simple Linear layer"""
    def __init__(self, in_dim: int = 128, out_dim: int = 992) -> None:
        super(Net, self).__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        return self.model(x)


def load_data() -> tuple[torch.utils.data.DataLoader, dict]:
    """Load face embeddings"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "embeddings", "embeddings.pkl")
    with open(pkl_path, "rb") as fp:  
        imgs_ndarray = np.stack(pickle.load(fp))

    # imgs_ndarray = np.random.rand(992, 128)
    img_tensor = torch.tensor(imgs_ndarray, dtype=torch.float32)
    labels = torch.arange(0, img_tensor.size(0))
    trainloader = DataLoader(TensorDataset(img_tensor, labels), batch_size=32, shuffle=True)
    num_examples = {"trainset": len(labels), "testset": len(labels)}
    return (trainloader, num_examples)


def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
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

def test(net: Net, testloader: DataLoader, DEVICE: torch.device) -> tuple[float, float]:
    """Validate the model on the test set."""
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
    trainloader, num_examples = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net, trainloader, 100, DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net, trainloader, DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)