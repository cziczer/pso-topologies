import numpy as np


class DefaultOperatorsMixin:
    """Order crossover (OX) + swap mutation operators.

    This is the baseline operator strategy ported from the original PSO implementation.
    Future operator variants (e.g. Operators1Mixin) should inherit from this class and
    override ``crossover`` and/or ``mutate`` to introduce different position-update
    strategies.

    Relies on ``self.num_city`` and ``self.dis_mat`` being set by ``PSOBase.__init__``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def crossover(self, current: list[int], guide: list[int]) -> tuple[list[int], float]:
        """Order crossover between *current* path and *guide* (neighbourhood best).

        A random slice is taken from *guide*; the remaining cities are appended in
        their original order from *current*.  Both orderings (remainder+slice and
        slice+remainder) are evaluated and the shorter one is returned.

        Returns:
            (new_path, length)
        """
        l = list(range(self.num_city))
        t = np.random.choice(l, 2, replace=False)
        x, y = int(min(t)), int(max(t))

        cross_part = guide[x:y]
        remainder = [c for c in current if c not in cross_part]

        candidate1 = remainder + cross_part
        l1 = self.path_length(candidate1)

        candidate2 = cross_part + remainder
        l2 = self.path_length(candidate2)

        if l1 <= l2:
            return candidate1, l1
        else:
            return candidate2, l2

    def mutate(self, path: list[int]) -> tuple[list[int], float]:
        """Swap mutation: randomly exchange two cities in *path*.

        Returns:
            (new_path, length)
        """
        path = path.copy()
        l = list(range(self.num_city))
        t = np.random.choice(l, 2, replace=False)
        x, y = int(min(t)), int(max(t))
        path[x], path[y] = path[y], path[x]
        return path, self.path_length(path)
