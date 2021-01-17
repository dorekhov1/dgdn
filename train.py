import math

import torch
import torchvision
import torch.nn.functional as F

from model import DGDN


def loss_func(x_input, x_recon, means, log_vars, etas):
    # Compute reconstruction loss
    recons_loss = F.mse_loss(x_input, x_recon)

    # Compute KL divergence for the Gaussian
    kld_loss_s = sum(
        [
            torch.mean(
                -0.5
                * torch.sum(1 + log_vars[i] - means[i] ** 2 - log_vars[i].exp(), dim=1),
                dim=0,
            )
            for i in range(len(means))
        ]
    )

    # Compute KL divergence for the Multinomial
    num_etas = len(etas)

    kld_loss_z = -math.log(num_etas) - (1 / num_etas) * (
        sum(
            [
                torch.mean(torch.sum(torch.log(etas[i]), dim=1), dim=0,)
                for i in range(num_etas)
            ]
        )
    )

    return (kld_loss_s + kld_loss_z) - recons_loss


def prepare_dataset(batch_size_train, batch_size_test):
    imagenet_data_train = torchvision.datasets.MNIST(
        "data/mnist/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    imagenet_data_test = torchvision.datasets.MNIST(
        "data/mnist/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    data_loader_train = torch.utils.data.DataLoader(
        imagenet_data_train, batch_size=batch_size_train, shuffle=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        imagenet_data_test, batch_size=batch_size_test, shuffle=True
    )

    return data_loader_train, data_loader_test


def train(epoch, network, train_loader, optimizer, log_interval):
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x_recon, means, log_vars, etas = network(data)
        loss = loss_func(data, x_recon, means, log_vars, etas)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
    return train_losses, train_counter


def test(network, test_loader):
    network.eval()
    test_loss = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            x_recon, means, log_vars, etas = network(data)
            test_loss += loss_func(data, x_recon, means, log_vars, etas)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print("\nTest set: Avg. loss: {:.4f}\n".format(test_loss))
    return test_losses


def main():
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.0002
    log_interval = 100

    random_seed = 1
    torch.manual_seed(random_seed)

    data_loader_train, data_loader_test = prepare_dataset(
        batch_size_train, batch_size_test
    )

    network = DGDN()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    test(network, data_loader_test)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, data_loader_train, optimizer, log_interval)
        test(network, data_loader_test)


if __name__ == "__main__":
    main()
