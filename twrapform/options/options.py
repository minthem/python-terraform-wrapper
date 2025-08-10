from __future__ import annotations

import copy
from abc import ABCMeta
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Type

from .opt_name import underscore_to_hyphen
from .opt_value import (
    backend_config_option,
    bool_flag_option,
    bool_option,
    list_str_option,
    var_option,
)
from .types import FrozenDict

_TF_OPTION_METANAME = "tf_options"
_TF_OPTION_DESCRIPTION_NAME = "tf_option_description"


@dataclass(frozen=True)
class FlagOption:
    """Metadata for Terraform-style flag options.

    Encapsulates how a dataclass field name and its value are translated into
    Terraform CLI arguments.

    Attributes:
        opt_name_func: Callable that maps a Python field name (e.g. "no_color")
            to a CLI option name (e.g. "-no-color").
        opt_values_func: Callable that converts an option name and a Python value
            into a tuple of CLI arguments. It must return an empty tuple when the
            value should not produce arguments (e.g. None).
    """

    opt_name_func: Callable[[str], str]
    opt_values_func: Callable[[str, Any], tuple[str, ...]]

    def get_opt_name(self, name: str) -> str:
        """Return the CLI option name for a Python field name.

        Args:
            name: Dataclass field name.

        Returns:
            A Terraform CLI option name (typically hyphenated).
        """
        return self.opt_name_func(name)

    def get_tf_option(self, name: str, values: Any) -> tuple[str, ...]:
        """Convert a field and its value into CLI arguments.

        Delegates to opt_values_func, after transforming the field name with
        opt_name_func. Implementation should return an empty tuple if the value
        results in no arguments.

        Args:
            name: Dataclass field name.
            values: Field value.

        Returns:
            Tuple of CLI arguments (possibly empty).
        """
        return self.opt_values_func(self.get_opt_name(name), values)


class PositionalOption:
    """Marker metadata for positional CLI arguments (no flag name)."""

    pass


UNDERSCORE_BOOL_OPTION_META = FlagOption(
    opt_name_func=underscore_to_hyphen, opt_values_func=bool_option
)

UNDERSCORE_BOOL_FLAG_OPTION_META = FlagOption(
    opt_name_func=underscore_to_hyphen, opt_values_func=bool_flag_option
)


@dataclass(frozen=True)
class TFCommandOptions(metaclass=ABCMeta):
    def __post_init__(self):
        """Freeze mutable mapping fields.

        Converts dict fields to FrozenDict via deep copy to preserve immutability
        guarantees of a frozen dataclass.
        """
        for field_info in fields(self):
            field_value = getattr(self, field_info.name)

            if field_value is None:
                continue

            # Convert dict to MappingProxyType
            if isinstance(field_value, dict):
                cp_field_value = copy.deepcopy(field_value)
                object.__setattr__(self, field_info.name, FrozenDict(cp_field_value))

    def convert_command_args(self) -> tuple[str, ...]:
        """Build the Terraform CLI arguments represented by this instance.

        - Starts with the command tuple provided by the command property.
        - Iterates all fields sorted by name and inspects their metadata:
          * FlagOption fields are converted to flag/value arguments.
          * PositionalOption fields are collected and appended last, in the same
            sorted-field order.
          * Fields without recognized metadata are ignored.

        Returns:
            Tuple of CLI arguments suitable for invocation.
        """
        result = self.command
        positional_args = tuple()
        sorted_fields = sorted(fields(self), key=lambda f: f.name)

        for _field in sorted_fields:
            tf_meta: FlagOption = _field.metadata.get(_TF_OPTION_METANAME)
            if tf_meta is None:
                continue

            if isinstance(tf_meta, FlagOption):
                result += tf_meta.get_tf_option(_field.name, getattr(self, _field.name))
            elif isinstance(tf_meta, PositionalOption):
                positional_args += (getattr(self, _field.name),)
            else:
                # Unsupported metadata type, skip it.
                pass

        result += positional_args

        return result

    @property
    def command(self) -> tuple[str, ...]:
        """Return the Terraform command/subcommand tokens (e.g. ("plan",))."""
        raise NotImplementedError("command_type must be implemented by subclasses")

    def convert_option(
        self,
        target_class: Type[TFCommandOptions],
        **override_values,
    ) -> TFCommandOptions:
        """Create a new options object of another type.

        Copies values for fields common to both source and target (where target
        fields have init=True). Explicit overrides take precedence.

        Args:
            target_class: Target dataclass type to instantiate.
            **override_values: Field values to override in the target.

        Returns:
            A new instance of target_class populated from this instance.
        """
        source_fields = {f.name for f in fields(self)}
        target_fields = {f.name for f in fields(target_class) if f.init}

        common_fields = source_fields & target_fields
        target_kwargs = {f: getattr(self, f) for f in common_fields}
        target_kwargs |= {
            k: v for k, v in override_values.items() if k in target_fields
        }

        return target_class(**target_kwargs)


@dataclass(frozen=True)
class OutputOptions:
    """Options that affect Terraform output formatting.

    Fields correspond to Terraform CLI flags related to output.
    """

    json: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, emit JSON output (if supported by the command). "
                "If None, omit the flag."
            ),
        },
    )

    no_color: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, disable ANSI colors. If None, omit the flag. "
            ),
        },
    )


@dataclass(frozen=True)
class LockOptions:
    """State locking options for Terraform commands."""

    lock: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, attempt to acquire a state lock. If False, skip locking. "
                "If None, omit the flag."
            ),
        },
    )

    lock_timeout: int | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=lambda n, v: (
                    (f"{n}={str(v)}s",) if v is not None else ()
                ),
            ),
            _TF_OPTION_DESCRIPTION_NAME: (
                "Max time in seconds to retry acquiring the lock. If None, "
                'omit the flag. Rendered as "-lock-timeout=<N>s"'
            ),
        },
    )


@dataclass(frozen=True)
class InputOptions:
    """Options that control interactive input."""

    input: bool | None = field(
        init=False,
        default=False,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If False (default), run non-interactively. This field is not"
                "settable via constructor (init=False)."
            ),
        },
    )


@dataclass(frozen=True)
class InitTaskOptions(OutputOptions, LockOptions, InputOptions, TFCommandOptions):
    """{{ InitTaskOptions_DOCSTRING }}"""

    backend: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, initialize the configured backend. If False, skip it. If None, omit the flag."
            ),
        },
    )

    backend_config: str | FrozenDict | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=backend_config_option,
            ),
            _TF_OPTION_DESCRIPTION_NAME: (
                "Backend configuration, as a file path string or key/value mapping. If None, omit the flag."
            ),
        },
    )

    upgrade: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, upgrade providers/modules to the latest available. If None, omit the flag."
            ),
        },
    )

    # NOTE Exclude because some versions are not supported
    json: bool | None = field(
        init=False,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "Excluded for compatibility with older Terraform versions."
            ),
        },
    )

    @property
    def command(self) -> tuple[str, ...]:
        """Return the command tuple for this task."""
        return ("init",)


@dataclass(frozen=True)
class PlanApplyOptionBase(LockOptions, OutputOptions, InputOptions):
    """Common options for "terraform plan" and "terraform apply"."""

    destroy: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, plan a full destroy. If None, omit the flag."
            ),
        },
    )

    target: tuple[str, ...] | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=list_str_option,
            ),
            _TF_OPTION_DESCRIPTION_NAME: (
                "Resource addresses to limit the operation to. If None, omit."
            ),
        },
    )

    var: dict | FrozenDict | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=var_option,
            ),
            _TF_OPTION_DESCRIPTION_NAME: (
                "Input variables as a dict/FrozenDict. If None, omit."
            ),
        },
    )

    var_file: tuple[str, ...] | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=list_str_option,
            ),
            _TF_OPTION_DESCRIPTION_NAME: (
                "One or more variable file paths. If None, omit."
            ),
        },
    )

    parallelism: int | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=lambda n, v: (
                    (f"{n}={str(v)}",) if v is not None else ()
                ),
            ),
            _TF_OPTION_DESCRIPTION_NAME: "Max concurrent operations. If None, omit.",
        },
    )


@dataclass(frozen=True)
class PlanTaskOptions(PlanApplyOptionBase, TFCommandOptions):
    """{{ PlanTaskOptions_DOCSTRING }}"""

    @property
    def command(self) -> tuple[str, ...]:
        """Return the command tuple for this task."""
        return ("plan",)


@dataclass(frozen=True)
class ApplyTaskOptions(PlanApplyOptionBase, TFCommandOptions):
    """{{ ApplyTaskOptions_DOCSTRING }}"""

    auto_approve: bool = field(
        init=False,
        default=True,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: "If True, skip interactive approval. Fixed to True (init=False).",
        },
    )

    @property
    def command(self) -> tuple[str, ...]:
        """Return the command tuple for this task."""
        return ("apply",)


@dataclass(frozen=True)
class OutputTaskOptions(OutputOptions, TFCommandOptions):
    """{{ OutputTaskOptions_DOCSTRING }}"""

    @property
    def command(self) -> tuple[str, ...]:
        """Return the command tuple for this task."""
        return ("output",)


@dataclass(frozen=True)
class WorkspaceSelectTaskOptions(TFCommandOptions):
    """{{ WorkspaceSelectTaskOptions_DOCSTRING }}"""

    workspace: str = field(
        init=True,
        metadata={
            _TF_OPTION_METANAME: PositionalOption(),
            _TF_OPTION_DESCRIPTION_NAME: "Workspace name to select.",
        },
    )

    or_create: bool | None = field(
        init=True,
        default=None,
        metadata={
            _TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META,
            _TF_OPTION_DESCRIPTION_NAME: (
                "If True, create the workspace if it does not exist. If None, omit the flag."
            ),
        },
    )

    @property
    def command(self) -> tuple[str, ...]:
        """Return the command tuple for this task."""
        return "workspace", "select"
