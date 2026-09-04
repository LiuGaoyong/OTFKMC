from pathlib import Path
from tempfile import TemporaryDirectory

from hydra import compose, initialize

from otfkmc.abc import FirstStep
from otfkmc.config import CONFIG_DIR

this_dir = Path(__file__).parent


def test_first_step(config: str) -> None:

    with TemporaryDirectory(dir=this_dir) as tmp:
        Path(tmp).mkdir(exist_ok=True, parents=True)
        print(f"Test in the temporary folder: '{tmp}'")
        with open(Path(tmp).joinpath("config.yaml"), "w") as f:
            f.write(config)

        with initialize(
            config_path=Path(tmp).relative_to(this_dir).as_posix(),
            job_name="first_step",
        ):
            print(list(Path(tmp).glob("*")))
            cfg = compose(
                config_name="config",
                # overrides=["db=mysql", "db.user=me"],
            )
            # HydraConfig.instance().set_config(cfg)
            obj = FirstStep(config=cfg)  # type: ignore
            for cluster in obj.system2cluster(obj.atoms2system(None)):
                obj.logger.info(f"{cluster.hash} {cluster}")


def test_config() -> None:
    with TemporaryDirectory(dir=this_dir) as tmp:
        Path(tmp).mkdir(exist_ok=True, parents=True)
        print(f"Test in the temporary folder: '{tmp}'")

        with initialize(
            config_path=Path(CONFIG_DIR)
            .relative_to(this_dir, walk_up=True)
            .as_posix(),
            job_name="first_step",
        ):
            print(list(Path(tmp).glob("*")))
            cfg = compose(
                config_name="run",
                overrides=[
                    "calculator=emt",
                    "atoms=octahedron",
                ],
            )
            # HydraConfig.instance().set_config(cfg)
            obj = FirstStep(config=cfg)  # type: ignore
            for cluster in obj.system2cluster(obj.atoms2system(None)):
                obj.logger.info(f"{cluster.hash} {cluster}")
