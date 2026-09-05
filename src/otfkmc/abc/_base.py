import os
import sys
from pathlib import Path
from typing import Literal

import hydra
from igraph import Graph
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from otfkmc.config import Config


class Base:
    """The base class for all classes.

    It provides:
        1. basic configuration (omegaconf.DictConfig)
        2. output directory (pathlib.Path)
    """

    def __init__(self, *, config: Config) -> None:
        assert isinstance(config, DictConfig | Config)
        self.config: Config = config
        self.path = Path(config.outputs)
        self.path.mkdir(parents=True, exist_ok=True)
        for k in ["minima", "gas", "ts"]:
            (self.path / k).mkdir(parents=True, exist_ok=True)
        self.network_path = self.path / "network.lgl"
        self.network = Graph(directed=False)

        loglevel = str(config.loglevel).upper()
        try:
            hydracfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
            outlogfile = str(hydracfg.job_logging.handlers.file.filename)
        except Exception:
            outlogfile = hydracfg = None
        outlogfile = config.logfile
        assert outlogfile is not None
        outlogfile = str(outlogfile)

        self.logger = log = Logger(
            core=Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        )
        log.add(sys.stderr, level=loglevel)
        logname = Path(outlogfile).name
        if logname != "-":
            logfile = self.path.joinpath(logname)
            log.add(logfile, level=loglevel)

        if hydracfg is not None:
            output_dir = hydracfg.runtime.output_dir
        else:
            output_dir = config.outputs
        output_dir = Path(output_dir).absolute()

        log.info("=" * 64)
        log.info("The Configuration:\n" + OmegaConf.to_yaml(config))
        log.info(f"Working floder   : {os.getcwd()}")
        log.info(f"self.path floder : {self.path}")
        log.info(f"Output floder    : {output_dir}")
        log.info(f"Output logfile   : {outlogfile}")
        log.info(f"Output loglevel  : {loglevel.upper()}")
        log.info("=" * 64)

        # check something
        parallel = str(config.parallel).lower()
        if hydracfg is not None and hydracfg.mode == "MULTIRUN":
            if parallel != "serial":
                raise ValueError(
                    "Please delete '--multirun,-m' option "
                    "when running this script. The multirun "
                    "mode is not supported because this program "
                    f"will be parallelized by '{parallel}' innerly."
                )
        assert parallel in ["serial", "joblib", "ray"], (
            f"Invalid parallel mode: {parallel}. Please "
            "choose one from 'serial', 'joblib', or 'ray'."
        )
        self.pmode: Literal["serail", "joblib", "ray"] = parallel  # type: ignore
