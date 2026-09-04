import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from otfkmc.config import Config

log = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"
config_dir = Path(__file__).parent.parent / "config"


@hydra.main(
    version_base=None,
    config_name="run",
    config_path=config_dir.as_posix(),
)
def run(cfg: Config) -> None:  # noqa: D103
    log.info("=" * 64)
    log.info("The Configuration:\n" + OmegaConf.to_yaml(cfg))
    assert isinstance(cfg, DictConfig)
    log.info(f"Working directory : {os.getcwd()}")
    hydracfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    outlogfile = hydracfg.job_logging.handlers.file.filename
    log.info(f"Output directory  : {hydracfg.runtime.output_dir}")
    log.info(f"Output logfile    : {outlogfile}")
    log.info("=" * 64)

    # check something
    if hydracfg.mode == "MULTIRUN":
        parallel = str(cfg.parallel).lower()
        if parallel != "serial":
            raise ValueError(
                "Please delete '--multirun,-m' option "
                "when running this script. The multirun "
                "mode is not supported because this program "
                f"will be parallelized by '{parallel}' innerly."
            )


if __name__ == "__main__":
    run()
