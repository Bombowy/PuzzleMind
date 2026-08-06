"""Cats rule-catalog boundary; no puzzle rules are implemented in v0.1."""

from abc import ABC, abstractmethod

from logicforge.rules.base_rule import BaseRule


class CatsRuleCatalog(ABC):
    """Define construction of the ordered Cats rule set without embedding rules.

    TODO: Model Cats constraints as small stateless rule classes in v0.5, with one
    responsibility per rule and exhaustive fixture-driven tests.
    """

    @abstractmethod
    def create(self) -> tuple[BaseRule, ...]:
        """Create the future deterministic Cats rule sequence.

        TODO: Implement catalog composition only after the generic rule engine has
        stable ordering, conflict, and explanation contracts in v0.4.
        """

        raise NotImplementedError
