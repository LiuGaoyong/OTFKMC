"""The module of reaction exploration."""

from typing import override

from graphatoms.system import Cluster, Gas  # type: ignore

from otfkmc.exploration._1Surf import SecondStepSurface as Surf
from otfkmc.exploration._2Ads import SecondStepAds as Ads
from otfkmc.exploration._3Bulk import SecondStepBulk as Bulk


class Exploration(Ads, Bulk, Surf):
    """The class for exploring the surface process."""

    @override
    def _explore_serial(self, cluster: Cluster, gas: Gas | None = None) -> None:
        Ads._explore_serial(self, cluster, gas)
        Bulk._explore_serial(self, cluster, gas)
        Surf._explore_serial(self, cluster, gas)

    @override
    def _explore_ray(self, cluster: Cluster, gas: Gas | None = None) -> None:
        Ads._explore_ray(self, cluster, gas)
        Bulk._explore_ray(self, cluster, gas)
        Surf._explore_ray(self, cluster, gas)
