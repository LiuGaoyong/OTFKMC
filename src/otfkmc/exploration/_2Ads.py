"""The class for exploring the adsorption process.

1. Got the site core for the adsorption process.
2. Try adsorption
"""

import itertools
from typing import override

from graphatoms.system import Cluster, Gas  # type: ignore

from ._0base import ExplBaseABC as SecondStepABC
from ._helper import helper_adsorption


class SecondStepAds(SecondStepABC):
    """The class for exploring the adsorption process."""

    @override
    def _explore_serial(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the adsorption process."""
        return
        result = itertools.starmap(
            helper_adsorption,
            itertools.product(
                [
                    # self.handle_cluster_by_core(core, system=system)
                    # for core in system.get_site_core(max_ncore=3)
                ],
                self.gas_lst,
            ),
        )
        list(result)

    @override
    def _explore_ray(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the adsorption process in ray mode."""
        return
        raise NotImplementedError("Ray mode is not supported.")
