"""Component registry for dynamic instantiation."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Registry for components."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}

    def register(self, name: str) -> callable:
        """Decorator to register a class.

        Args:
            name: Name to register under.

        Returns:
            Decorator function.
        """

        def decorator(cls: type[T]) -> type[T]:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        """Get registered class by name.

        Args:
            name: Registered name.

        Returns:
            The registered class.

        Raises:
            KeyError: If name not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def instantiate(self, name: str, **kwargs: Any) -> Any:
        """Instantiate registered class with kwargs.

        Args:
            name: Registered name.
            **kwargs: Arguments for instantiation.

        Returns:
            Instantiated object.
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list_components(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of component names.
        """
        return list(self._registry.keys())


# Global registries
tactile_encoders = Registry()
dynamics_models = Registry()
gating_models = Registry()
planners = Registry()
policies = Registry()
