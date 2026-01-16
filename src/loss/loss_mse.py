from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_depth_mask: Tensor | None = None,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        if (
            valid_depth_mask is not None
            and valid_depth_mask.max() > 0.5
            and valid_depth_mask.min() < 0.5
        ):
            delta = delta[~valid_depth_mask]
        return self.cfg.weight * (delta**2).mean()
