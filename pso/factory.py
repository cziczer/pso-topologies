from __future__ import annotations

from data.loader import TSPInstance
from pso.base import PSOBase
from pso.operators.default import DefaultOperatorsMixin
from pso.operators.cognitive import CognitiveMixin
from topologies.global_ import GlobalTopologyMixin
from topologies.ring import RingTopologyMixin
from topologies.tree import TreeTopologyMixin
from topologies.mesh import MeshTopologyMixin
from topologies.torus import TorusTopologyMixin
from topologies.free_scale import FreeScaleTopologyMixin


# --------------------------------------------------------------------------
# Registry maps — add new variants here as they are implemented
# --------------------------------------------------------------------------

_OPERATOR_MAP: dict[str, type] = {
    "default":   DefaultOperatorsMixin,
    "cognitive": CognitiveMixin,
    # "operators1": Operators1Mixin,  # add when the variant is implemented
}

_TOPOLOGY_MAP: dict[str, type] = {
    "global":     GlobalTopologyMixin,
    "ring":       RingTopologyMixin,
    "tree":       TreeTopologyMixin,
    "mesh":       MeshTopologyMixin,
    "torus":      TorusTopologyMixin,
    "freescale":  FreeScaleTopologyMixin,
}


class AlgorithmFactory:
    """Parse algorithm name strings and assemble composed PSO classes.

    Name format::

        PSO[-<OperatorVariant>][-<Topology>]

    Examples::

        "PSO"                  -> default operators + global topology
        "PSO-Ring"             -> default operators + ring topology
        "PSO-Torus"            -> default operators + torus topology
        "PSO-Operators1-Ring"  -> Operators1 operators + ring topology  (future)

    Token matching is case-insensitive.  Operator and topology token namespaces
    are disjoint, so ordering of the two optional tokens does not matter.
    """

    @staticmethod
    def parse(name: str) -> tuple[str, str, str]:
        """Parse *name* into (base, operator_key, topology_key).

        Raises:
            ValueError: if the first token is not ``"PSO"`` or an unknown token
                is encountered.
        """
        tokens = name.split("-")
        if tokens[0].upper() != "PSO":
            raise ValueError(
                f"Algorithm name must start with 'PSO', got: '{tokens[0]}'"
            )

        operator_key = "default"
        topology_key = "global"

        for token in tokens[1:]:
            lower = token.lower()
            if lower in _OPERATOR_MAP:
                operator_key = lower
            elif lower in _TOPOLOGY_MAP:
                topology_key = lower
            else:
                raise ValueError(
                    f"Unknown token '{token}' in algorithm name '{name}'.\n"
                    f"  Known operator variants : {sorted(_OPERATOR_MAP)}\n"
                    f"  Known topologies        : {sorted(_TOPOLOGY_MAP)}"
                )

        return "PSO", operator_key, topology_key

    @staticmethod
    def build(name: str, instance: TSPInstance, **kwargs) -> PSOBase:
        """Assemble and instantiate the composed PSO class for *name*.

        MRO of the composed class:
            (OperatorMixin, TopologyMixin, PSOBase)

        Operator methods shadow the abstract stubs in ``PSOBase``; topology
        methods shadow the no-op defaults.

        Args:
            name: Algorithm name string (e.g. ``"PSO-Ring"``).
            instance: Loaded TSP problem instance.
            **kwargs: Forwarded to the composed class constructor.  Each mixin
                extracts the kwargs it recognises and passes the rest up.

        Returns:
            A fully initialised ``PSOBase`` subclass instance.
        """
        _, op_key, topo_key = AlgorithmFactory.parse(name)
        op_cls   = _OPERATOR_MAP[op_key]
        topo_cls = _TOPOLOGY_MAP[topo_key]

        composed_cls = type(name, (op_cls, topo_cls, PSOBase), {})
        return composed_cls(instance=instance, **kwargs)
