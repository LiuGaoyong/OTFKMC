"""The module of reaction exploration."""

from typing import override

from graphatoms.system import System

from otfkmc.exploration._1Surf import SecondStepSurface as Surf
from otfkmc.exploration._2Ads import SecondStepAds as Ads
from otfkmc.exploration._3Bulk import SecondStepBulk as Bulk


class Exploration(Ads, Bulk, Surf):
    """The class for exploring the surface process."""

    @override
    def _explore_serial(self, system: System) -> None:
        Ads._explore_serial(self, system)
        Bulk._explore_serial(self, system)
        Surf._explore_serial(self, system)

    @override
    def _explore_ray(self, system: System) -> None:
        Ads._explore_ray(self, system)
        Bulk._explore_ray(self, system)
        Surf._explore_ray(self, system)
