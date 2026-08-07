"""Optional catalog boundary alongside the production Cats rule loop."""

from abc import ABC, abstractmethod

from logicforge.rules.base_rule import BaseRule


class CatsRuleCatalog(ABC):
    """Define a generic catalog extension point without owning rule behavior."""

    @abstractmethod
    def create(self) -> tuple[BaseRule, ...]:
        """Create the catalog-provided deterministic Cats rule sequence."""

        raise NotImplementedError
