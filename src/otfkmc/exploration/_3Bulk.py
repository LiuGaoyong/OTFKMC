import itertools
from typing import override

from graphatoms.system import Cluster, System

from otfkmc.exploration._2expl_0_Base import SecondStepABC
from otfkmc.runner._1parser import FirstStep


class SecondStepBulk(SecondStepABC):
    """The class for exploring the bulk process."""

    @override
    def _explore_serial(self, system: System) -> None:
        """Explore the bulk process."""
        result = itertools.starmap(
            helper_neb,
            itertools.product(
                [self],
                [
                    self.handle_cluster_by_core(core, system=system)
                    for core in system.get_site_core(max_ncore=3)
                ],
            ),
        )
        list(result)

    @override
    def _explore_ray(self, system: System) -> None:
        """Explore the bulk process in ray mode."""
        return
        raise NotImplementedError("Ray mode is not supported.")


def helper_neb(self: FirstStep, cluster: Cluster) -> None:
    """Helper function for neb process."""
    return
    raise NotImplementedError("Not implemented.")
