import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


def train(
    net: Net,
    trainset: Dataset,
    epochs: int,
    device: torch.device,
    id: str = "",
    log_tensorboard: bool = False
) -> None:
    """Train the network."""
    # Initial tensorboard SummaryWriter
    if log_tensorboard:
        writer = SummaryWriter()

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
        net.train()
        for i, (img_batch, labels) in enumerate(trainloader):
            img_batch, labels = img_batch.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Log loss and accuracy to tensorboard
        if log_tensorboard:
            loss, accuracy = test(net, trainset, device)
            loss_label = "Loss/whole" if id == "" else f"Loss/partition{id}"
            acc_label = "Accuracy/whole" if id == "" else f"Accuracy/partition{id}"
            writer.add_scalar(loss_label, loss, epoch)
            writer.add_scalar(acc_label, accuracy, epoch)
   

def test(
    net: Net, 
    testset: Dataset, 
    device: torch.device
) -> tuple[float, float]:
    """Validate the model on the test set."""
    net.eval()
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    # correct, total, loss = 0, 0, 0.0
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            # total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return (loss / len(testset), correct / len(testset))
    # return (loss / len(testloader.dataset), correct / total)

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--log_tensorboard",
        action="store_true",
        help="Whether to log loss and metrics to Tensorboard",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=30,
        required=False,
        help="Number of training epochs for each round",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainset, testset, num_examples = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net, trainset, args.train_epochs, DEVICE, log_tensorboard=args.log_tensorboard)
    print("Evaluate model")
    loss, accuracy = test(net, testset, DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)