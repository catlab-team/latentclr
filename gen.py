import hydra
from omegaconf import DictConfig

import colat.runner as runner
import colat.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="gen")
def gen(cfg: DictConfig):
    utils.display_config(cfg)
    runner.generate(cfg)


if __name__ == "__main__":
    gen()
