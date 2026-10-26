"""Definitions of unprivileged demographic groups for fairness analysis."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class UnprivilegedGroups(ABC):
    """
    Abstract base class for unprivileged-group definitions.

    Subclass this and implement get_groups() to declare which group within
    each protected attribute should be treated as historically disadvantaged
    by the fairness auditor.

    Return format of get_groups()
    ------------------------------
    A dict mapping attribute name → unprivileged group value(s).

    * Single value  → always treated as unprivileged:
        {"attribute": value_x}
    * List of values → auditor combines them into one unprivileged group:
        {"attribute": [value_x, value_y, value_z]}

    Example
    -------
    ::
        class MyGroups(UnprivilegedGroups):
            def get_groups(self) -> dict:
                return {
                    "attribute_a": value_x,
                    "attribute_b": [value_x, value_y, value_z],
                    ...
                }
    """

    @abstractmethod
    def get_groups(self) -> Dict[str, Any]:
        """Return mapping of protected attribute name → unprivileged group value(s)."""

    def __call__(self) -> Dict[str, Any]:
        """Delegate to get_groups() so instances are directly callable."""
        return self.get_groups()


class AdultIncomeGroups(UnprivilegedGroups):
    """
    Unprivileged group definitions for this Adult Income Dataset
    
    TODO: Select protected attributes and unpriviledged groups (values) for each attribute.
          All values that are not declared as unpriviledged are automatically deemed as priviledged.


    """

    def get_groups(self) -> Dict[str, Any]:
        raise NotImplementedError("TODO")
