from colat.projectors import Projector
from colat.utils.net_utils import create_mlp


class LinearProjector(Projector):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        batchnorm: bool = True,
        normalize: bool = True,
    ):
        # linear layer
        net = create_mlp(
            depth=1,
            in_features=in_features,
            middle_features=-1,
            out_features=out_features,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=batchnorm,
        )
        super().__init__(net, normalize)
