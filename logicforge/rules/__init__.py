"""Puzzle-rule contracts and rule-engine orchestration boundaries."""

from logicforge.rules.base_rule import BaseRule, RuleOutcome
from logicforge.rules.rule_engine import RuleEngine

__all__ = ["BaseRule", "RuleEngine", "RuleOutcome"]
