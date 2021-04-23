import torch

from colat.models.abstract import Model


class Fixed(Model):
    """K global fixed directions"""

    def __init__(
        self, k: int, size: int, alpha: float = 0.1, normalize: bool = True
    ) -> None:
        super().__init__(k=k, size=size, alpha=alpha, normalize=normalize)

        self.k = k
        self.size = size
        self.params = torch.nn.Parameter(torch.randn(k, size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z = z.repeat(
            (
                self.k,
                1,
                1,
            )
        )

        #  add directions
        z += torch.reshape(self.post_process(self.params), (self.k, 1, self.size))

        # reshape
        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.params)[k : k + 1, :]
