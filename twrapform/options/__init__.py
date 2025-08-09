from .options import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
    WorkspaceSelectTaskOptions,
)
from .types import FrozenDict

SupportedTerraformTask = (
    InitTaskOptions
    | PlanTaskOptions
    | ApplyTaskOptions
    | OutputTaskOptions
    | WorkspaceSelectTaskOptions
)

__all__ = [
    "ApplyTaskOptions",
    "InitTaskOptions",
    "OutputTaskOptions",
    "PlanTaskOptions",
    "WorkspaceSelectTaskOptions",
    "SupportedTerraformTask",
    "FrozenDict",
]
