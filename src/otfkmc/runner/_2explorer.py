from typing import override

from graphatoms.system import System

from otfkmc.runner._2expl_1_Surf import SecondStepSurface as Surf
from otfkmc.runner._2expl_2_Ads import SecondStepAds as Ads
from otfkmc.runner._2expl_3_Bulk import SecondStepBulk as Bulk


class SecondStep(Ads, Bulk, Surf):
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
