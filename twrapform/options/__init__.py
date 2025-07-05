from .options import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
    WorkspaceSelectTaskOptions,
)

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
]
