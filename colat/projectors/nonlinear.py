from colat.projectors import Projector
from colat.utils.net_utils import create_mlp


class NonlinearProjector(Projector):
    def __init__(
        self,
        depth: int,
        in_features: int,
        middle_features: int,
        out_features: int,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
        normalize: bool = True,
    ):
        # nonlinear mlp
        net = create_mlp(
            depth=depth,
            in_features=in_features,
            middle_features=middle_features,
            out_features=out_features,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=final_norm,
        )
        super().__init__(net, normalize)
