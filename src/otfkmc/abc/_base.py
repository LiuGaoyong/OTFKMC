import os
import sys
from pathlib import Path
from typing import Literal

import hydra
from ase.symbols import Symbols
from graphatoms.system import Cluster, Gas
from igraph import Graph
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf


class Base:
    """The base class for all classes.

    It provides:
        1. basic configuration (omegaconf.DictConfig)
        2. output directory (pathlib.Path)
    """

    def __init__(self, *, config: DictConfig) -> None:
        assert isinstance(config, DictConfig)
        self.config: DictConfig = config
        self.path = Path(config.output)
        self.path.mkdir(parents=True, exist_ok=True)
        for k in ["minima", "gas", "ts"]:
            (self.path / k).mkdir(parents=True, exist_ok=True)
        self.network_path = self.path / "network.lgl"
        self.network = Graph(directed=False)

        loglevel = str(config.get("loglevel", "DEBUG"))
        try:
            hydracfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
            outlogfile = str(hydracfg.job_logging.handlers.file.filename)
        except Exception:
            outlogfile = hydracfg = None
        outlogfile = config.get("logfile", outlogfile)
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

        log.info("=" * 64)
        log.info("The Configuration:\n" + OmegaConf.to_yaml(config))
        log.info(f"Working directory : {os.getcwd()}")
        if hydracfg is not None:
            log.info(f"Output directory  : {hydracfg.runtime.output_dir}")
        log.info(f"Output logfile    : {outlogfile}")
        log.info(f"Output loglevel   : {loglevel.upper()}")
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

    def cluster_path(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> Path:
        """Get the path to the cluster."""
        if isinstance(cluster, Gas):
            type = "gas"
        else:
            assert isinstance(cluster, Cluster)
            assert type in ("minima", "ts")
        symbols: Symbols = cluster.symbols
        fml: str = symbols.get_chemical_formula("metal")
        p = self.path / type / fml
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        return p / f"{cluster.hash}.npz"

    def cluster_save(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> None:
        """Save the cluster."""
        cluster.write_npz(self.cluster_path(cluster=cluster, type=type))

    def cluster_exists(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> bool:
        """Check if the cluster exists."""
        return self.cluster_path(cluster=cluster, type=type).exists()

    def cluster_check(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> bool:
        event: DictConfig = self.config.event
        fmax = float(event.get("max_force", 0.05))
        if type == "ts":
            assert isinstance(cluster, Cluster)
            mfreq_ts = float(event.get("min_frequency_for_ts", 50.0))
            return cluster.check_ts(fmax, mfreq_ts)
        elif isinstance(cluster, Gas) or type == "minima":
            mfreq_minima = float(event.get("min_frequency", 30.0))
            return cluster.check_minima(fmax, mfreq_minima)
        else:
            raise ValueError(
                f"Unknown type={type}, or type(cluster)="  #
                f"{cluster.__class__.__name__}"
            )
