"""Distribution registry for string-based creation."""

from typing import Type

from ..core.distribution import TimeEvolvingDistribution


class DistributionRegistry:
    """
    Registry for distribution implementations.

    Allows users to register custom distributions and retrieve
    them by name. Built-in distributions are registered automatically.

    Example:
        >>> # Get a built-in distribution
        >>> dist_class = DistributionRegistry.get("normal")
        >>> dist = dist_class()
        >>>
        >>> # Create instance directly
        >>> dist = DistributionRegistry.create("generalized_laplace")
        >>>
        >>> # Register a custom distribution
        >>> DistributionRegistry.register("my_dist", MyDistribution)
    """

    _registry: dict[str, Type[TimeEvolvingDistribution]] = {}  # type: ignore[type-arg]

    @classmethod
    def register(
        cls, name: str, distribution_class: Type[TimeEvolvingDistribution]  # type: ignore[type-arg]
    ) -> None:
        """
        Register a distribution class.

        Args:
            name: Name to register the distribution under (case-insensitive)
            distribution_class: The distribution class to register
        """
        cls._registry[name.lower()] = distribution_class

    @classmethod
    def get(cls, name: str) -> Type[TimeEvolvingDistribution] | None:  # type: ignore[type-arg]
        """
        Get a distribution class by name.

        Args:
            name: Distribution name (case-insensitive)

        Returns:
            Distribution class or None if not found
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all registered distribution names.

        Returns:
            List of distribution names
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str) -> TimeEvolvingDistribution:  # type: ignore[type-arg]
        """
        Create a distribution instance by name.

        Args:
            name: Distribution name (case-insensitive)

        Returns:
            Distribution instance

        Raises:
            ValueError: If distribution name is not registered
        """
        dist_class = cls.get(name)
        if dist_class is None:
            available = cls.list_available()
            raise ValueError(
                f"Unknown distribution: '{name}'. Available: {available}"
            )
        return dist_class()


def _register_builtins() -> None:
    """Register all built-in distributions."""
    from .generalized_laplace import GeneralizedLaplaceDistribution
    from .normal import NormalDistribution
    from .student_t import StudentTDistribution
    from .skew_normal import SkewNormalDistribution

    DistributionRegistry.register("generalized_laplace", GeneralizedLaplaceDistribution)
    DistributionRegistry.register("laplace", GeneralizedLaplaceDistribution)  # alias
    DistributionRegistry.register("normal", NormalDistribution)
    DistributionRegistry.register("gaussian", NormalDistribution)  # alias
    DistributionRegistry.register("student_t", StudentTDistribution)
    DistributionRegistry.register("t", StudentTDistribution)  # alias
    DistributionRegistry.register("skew_normal", SkewNormalDistribution)
    DistributionRegistry.register("skewnormal", SkewNormalDistribution)  # alias


# Auto-register built-ins on module import
_register_builtins()
