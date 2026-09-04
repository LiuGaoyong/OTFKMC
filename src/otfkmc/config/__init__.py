from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from otfkmc.config.atoms import (
    AseReadAtomsConfig,
    AtomsConfig,
    OctahedronAtomsConfig,
)
from otfkmc.config.bonds import (
    BondsConfig,
    RawBondsConfig,
)
from otfkmc.config.calculator import (
    CalcConfig,
    EMTCalcConfig,
    NequipCalcConfig,
)
from otfkmc.config.gas import (
    AseReadGasConfig,
    GasConfig,
)

CONFIG_DIR = Path(__file__).parent


@dataclass
class SiteConfig:
    method: str = "distance"  # or hop
    env_threshold: float = 15  # the environment threshold
    max_moved_threshold: float = 8  # the maximum moved threshold


@dataclass
class ExplConfig:
    maxtry: int = 1000
    maxconfidence: float = 10
    max_ncore_for_adsorption: int = 3  # 4, 5, 6
    allow_multiple_adsorption: bool = False
    allow_explore_bulk: bool = False
    thetacutoff: float = 30


@dataclass
class EventConfig:
    check_frequency: bool = True  # whether to check the frequency
    max_force: float = 0.05  # eV / Angstrom
    min_frequency: float = 30.0  # cm^-1
    min_frequency_for_ts: float = 20.0  # cm^-1


@dataclass
class OptConfig:
    method: str = "lbfgs"
    fmax: float = 0.05
    steps: int = 200


@dataclass
class Config:
    atoms: AtomsConfig = MISSING
    bonds: BondsConfig = MISSING
    system: Any = MISSING

    calculator: CalcConfig = MISSING
    gas: GasConfig = MISSING

    outputs: str = "./zzz"
    parallel: str = "serial"  # Literal["serial", "joblib", "ray"]
    loglevel: str = "info"  # Literal["debug", "info", "warning", "error"]
    logfile: str = "log.txt"
    restart: bool = True
    debug: bool = False

    site: SiteConfig = field(default_factory=SiteConfig)
    exploration: ExplConfig = field(default_factory=ExplConfig)
    event: EventConfig = field(default_factory=EventConfig)
    optimizer: OptConfig = field(default_factory=OptConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="atoms", name="atoms_aseread", node=AseReadAtomsConfig)
cs.store(group="atoms", name="atoms_octahedron", node=OctahedronAtomsConfig)
cs.store(group="bonds", name="bonds_raw", node=RawBondsConfig)
cs.store(group="calculator", name="calc_emt", node=EMTCalcConfig)
cs.store(group="calculator", name="calc_nequip", node=NequipCalcConfig)
cs.store(group="gas", name="gas_none", node=GasConfig)
cs.store(group="gas", name="gas_aseread", node=AseReadGasConfig)


@hydra.main(config_path=".", config_name="run", version_base=None)
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
