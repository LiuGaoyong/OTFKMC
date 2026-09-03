# ruff: noqa
from otfkmc.abc import FirstStep
from tempfile import TemporaryDirectory
from pathlib import Path
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig


def test_first_step(config: str) -> None:
    this_dir = Path(__file__).parent

    with TemporaryDirectory(dir=this_dir) as tmp:
        Path(tmp).mkdir(exist_ok=True, parents=True)
        print(f"Test in the temporary folder: '{tmp}'")
        with open(Path(tmp).joinpath("config.yaml"), "w") as f:
            f.write(config)

        with initialize(
            config_path=Path(tmp).relative_to(this_dir).as_posix(),
            job_name="first_step",
        ):
            print(Path(tmp).glob("*"))
            cfg = compose(
                config_name="config",
                # overrides=["db=mysql", "db.user=me"],
            )
            # HydraConfig.instance().set_config(cfg)
            obj = FirstStep(config=cfg)
            for cluster in obj.system2cluster(obj.atoms2system(None)):
                obj.logger.info(f"{cluster.hash} {cluster}")
