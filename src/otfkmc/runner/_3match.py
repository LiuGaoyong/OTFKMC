import itertools
from pathlib import Path

import numpy as np
from graphatoms.system import Cluster, System
from igraph import Graph

from otfkmc.abc import FirstStep


class ThirdStep:
    def match(self, system: System | None = None) -> None:
        """Matching of the catalyst elemental reaction."""
        if system is None:
            system = self.catalyst
        network = Graph.Read(self.network_path)
        print([network.vs[i]["name"] for e in network.es for i in e.tuple])

        parallel = str(self.config.parallel).lower()
        if parallel == "serial":
            result = itertools.starmap(
                helper_match,
                itertools.product(
                    [self],
                    [
                        network.vs[i]["name"]
                        for e in network.es
                        for i in e.tuple
                    ],
                    [system],
                ),
            )
            result = list(result)
            print(result)
        elif parallel == "ray":
            raise NotImplementedError("Ray mode is not supported.")
        else:
            raise ValueError(f"Parallel mode '{parallel}' is not supported.")


def helper_match(
    self: FirstStep,
    cluster_name: str,
    system: System,
) -> None | np.ndarray:
    """Helper function for matching process."""
    cluster = Cluster.read_npz(self.path / Path(cluster_name))
    result = system.get_match_mode(cluster)  # type: ignore
    assert not isinstance(result, int)
    return result
