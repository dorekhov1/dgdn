import torch
from torch import nn


class Pool(nn.Module):
    """Implementation of the stochastic pooling layer.

    Attributes:
        p_x (int): Size of the pooling window in the x-dimension.
        p_y (int): Size of the pooling window in the y-dimension.
        pooling_layer (nn.Sequential): Neural net that decides which pixel to keep when pooling.
    """

    def __init__(self, p_x, p_y):
        super(Pool, self).__init__()
        self.p_x = p_x
        self.p_y = p_y

        self.pooling_layer = nn.Sequential(
            nn.Linear(
                in_features=self.p_x * self.p_y,
                out_features=self.p_x * self.p_y,
                bias=False,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=self.p_x * self.p_y,
                out_features=self.p_x * self.p_y,
                bias=False,
            ),
            nn.Softmax(),
        )

    def forward(self, c):
        """Applies stochastic pooling to the input feature map.

        Args:
            c (Tensor): The input feature map

        Returns:
            pooled_data (Tensor): The output feature map with reduced size
            etas (Tensor): Vectors that define the Multinomial distribution, from which the pooling vectors are drawn

        """

        # for now, assume the that each dimension is evenly divisible by pooling size
        n = int(c.shape[-2] / self.p_x)
        m = int(c.shape[-1] / self.p_y)

        etas = []

        pooled_data = torch.empty(c.shape[0], c.shape[1], n, m, dtype=torch.float)

        for i in range(n):
            for j in range(m):
                tile = torch.flatten(
                    c[
                        :,
                        :,
                        i * self.p_x : (i + 1) * self.p_x,
                        j * self.p_y : (j + 1) * self.p_y,
                    ],
                    start_dim=2,
                )
                for k in range(tile.shape[1]):
                    tile_slice = tile[:, k, :]
                    eta = self.pooling_layer(tile_slice)
                    etas.append(eta)

                    zeta = torch.multinomial(eta, 1)
                    picked_entry = torch.flatten(tile_slice.gather(1, zeta))
                    pooled_data[:, k, i, j] = picked_entry

        return pooled_data, etas


class Unpool(nn.Module):
    """Implementation of the stochastic unpooling layer.

    Attributes:
        p_x (int): Size of the pooling window in the x-dimension.
        p_y (int): Size of the pooling window in the y-dimension.

    """

    def __init__(self, p_x, p_y):
        super(Unpool, self).__init__()
        self.p_x = p_x
        self.p_y = p_y

    def forward(self, s, etas):
        """Applies stochastic pooling to the input feature map.

        Args:
            s (Tensor): The input feature map
            etas (Tensor): Vectors that define the Multinomial distribution, from which the pooling vectors are drawn.

        Returns:
            unpooled_data(Tensor): The output feature map with an increased size

        """
        n = s.shape[-2]
        m = s.shape[-1]

        unpooled_data = torch.empty(
            s.shape[0], s.shape[1], self.p_x * n, self.p_y * m, dtype=torch.float
        )

        cnt = 0
        for i in range(n):
            for j in range(m):
                for k in range(s.shape[1]):

                    s_value = s[:, k, i, j]

                    eta = etas[cnt]
                    zeta = torch.multinomial(eta, 1)

                    tile = torch.zeros(
                        s.shape[0], self.p_x * self.p_y, dtype=torch.float
                    )

                    tile = tile.scatter_(1, zeta, s_value.reshape(*s_value.shape, 1))

                    unpooled_data[
                        :,
                        k,
                        i * self.p_x : (i + 1) * self.p_x,
                        j * self.p_y : (j + 1) * self.p_y,
                    ] = tile.resize(s.shape[0], self.p_x, self.p_y)

                    cnt += 1

        return unpooled_data


class DGDN(nn.Module):
    """Implementation of the Deep Generative Deconvolutional Network.

    Args:
        p_x (int): Size of the pooling window in the x-dimension for stochastic pooling/unpooling.
        p_y (int): Size of the pooling window in the y-dimension for stochastic pooling/unpooling.
        K_1 (int): Kernel size for the CNN in layer 1.
        K_2 (int): Kernel size for the CNN in layer 2.
        N_c (int): Number of channels in the input.

    Attributes:
          encoder_layer1 (nn.Conv2d): The first CNN layer of the encoder.
          pooling (Pool): The stochastic pooling layer for the encoder.
          encoder_layer2 (nn.Conv2d): The second CNN layer of the encoder.
          encoder_h (list): List of K_2 MLPs that are used as the first layer of the neural networks that
                        produce the mean and log-variance for the code generation.
          encoder_mean (list): List of K_2 MLP that are used as the second layer of the neural networks that
                        produce the mean for the code generation.
          encoder_logvar (list): List of K_2 MLP that are used as the second layer of the neural networks that
                        produce the log-variance for the code generation.
          decoder_layer2 (nn.Conv2d): The first CNN layer of the decoder.
          unpooling (Pool): The stochastic unpooling layer for the decoder.
          decoder_layer1 (nn.Conv2d): The second CNN layer of the decoder.
          alpha (nn.Parameter): Variance used to add Gaussian noise to the final result.
    """

    def __init__(
        self, p_x=3, p_y=3, K_1=30, K_2=80, N_c=1,
    ):
        super(DGDN, self).__init__()

        self.encoder_layer1 = nn.Conv2d(
            in_channels=N_c, out_channels=K_1, kernel_size=8, bias=False
        )
        self.pooling = Pool(p_x, p_y)
        self.encoder_layer2 = nn.Conv2d(
            in_channels=K_1, out_channels=K_2, kernel_size=4, bias=False
        )

        self.encoder_h = [
            nn.Sequential(nn.Linear(16, 16), nn.Tanh()) for _ in range(K_2)
        ]
        self.encoder_mean = [nn.Linear(16, 16) for _ in range(K_2)]
        self.encoder_logvar = [nn.Linear(16, 16) for _ in range(K_2)]

        self.decoder_layer2 = nn.ConvTranspose2d(
            in_channels=K_2, out_channels=K_1, kernel_size=4, bias=False
        )
        self.unpooling = Unpool(p_x, p_y)
        self.decoder_layer1 = nn.ConvTranspose2d(
            in_channels=K_1, out_channels=N_c, kernel_size=8, bias=False
        )

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def sample(self, log_var, mean, new_dim):
        """Samples from Normal distribution for each element in the batch.

        Args:
            log_var (Tensor): Log-variances for each element in the batch.
            mean (Tensor): Means for each element in the batch.
            new_dim (int): Dimension to use when resizing the result from 1D to 2D.

        Returns:
            d (Tensor): For each element of the batch, returns a new_dim X new_dim matrix,
                         with values sampled from the corresponding Normal distribution.

        """
        std = torch.exp(0.5 * log_var)
        d = torch.normal(mean, std).resize(mean.shape[0], new_dim, new_dim)
        return d

    def add_noise(self, s):
        """ Adds Gaussian noise to the generated data.

        Args:
            s (Tensor): The reconstructed data tensor.

        Returns:
            (Tensor): The reconstructed data tensor with extra Gaussian noise, defined by self.alpha.
        """
        return torch.normal(s, (1 / self.alpha) * torch.eye(s.shape[-2], s.shape[-1]))

    def forward(self, x):
        """Applies the Deep Generative Deconvolutional Network to the input of images.

        Args:
            x (Tensor): The batch of images.

        Returns:
            x (Tensor): The batch of reconstructed images.
            means (Tensor): The batch of means that were used for code generation.
            log_vars (Tensor): The batch of log-variances that were used for code generation.
            etas (Tensor): The vectors that were used to define the Multinomial distribution
                           for stochastic pooling/unpooling

        """
        c1 = self.encoder_layer1(x)
        c1, etas = self.pooling(c1)
        c2 = self.encoder_layer2(c1)

        log_vars = []
        means = []

        s = torch.empty(c2.shape, dtype=torch.float)
        for i in range(c2.shape[1]):
            h = self.encoder_h[i](torch.flatten(c2[:, i, :, :], start_dim=1))
            log_var = self.encoder_logvar[i](h)
            mean = self.encoder_mean[i](h)
            log_vars.append(log_var)
            means.append(mean)
            s[:, i, :, :] = self.sample(log_var, mean, c2.shape[-1])

        s2 = self.decoder_layer2(s)
        s1 = self.unpooling(s2, etas)
        s1 = self.decoder_layer1(s1)

        x = self.add_noise(s1)

        return x, means, log_vars, etas
