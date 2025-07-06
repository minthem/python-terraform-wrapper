from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Callable, Type

from .opt_name import underscore_to_hyphen
from .opt_value import (
    backend_config_option,
    bool_flag_option,
    bool_option,
    list_str_option,
    var_option,
)

TF_OPTION_METANAME = "tf_options"


@dataclass(frozen=True)
class FlagOption:
    """Metadata class for Terraform command-line options.

    This class encapsulates the logic for converting Python field names
    to Terraform command-line option names and values.

    Attributes:
        opt_name_func: Function that converts Python field names to option names
        opt_values_func: Function that converts field values to command-line arguments
    """

    opt_name_func: Callable[[str], str]
    opt_values_func: Callable[[str, Any], tuple[str, ...]]

    def get_opt_name(self, name: str) -> str:
        """Convert a Python field name to a Terraform option name.

        Args:
            name: The Python field name to convert

        Returns:
            The converted option name suitable for Terraform CLI
        """
        return self.opt_name_func(name)

    def get_tf_option(self, name: str, values: Any) -> tuple[str, ...]:
        """Generate Terraform command-line arguments from field name and value.

        Args:
            name: The field name
            values: The field value

        Returns:
            A tuple of command-line arguments for Terraform
        """
        return self.opt_values_func(self.get_opt_name(name), values)


class PositionalOption:
    """Metadata class for Terraform positional command-line arguments."""

    pass


UNDERSCORE_BOOL_OPTION_META = FlagOption(
    opt_name_func=underscore_to_hyphen, opt_values_func=bool_option
)

UNDERSCORE_BOOL_FLAG_OPTION_META = FlagOption(
    opt_name_func=underscore_to_hyphen, opt_values_func=bool_flag_option
)


@dataclass(frozen=True)
class TFCommandOptions:
    def __post_init__(self):
        """Post-initialization processing.

        Ensures that dictionary fields are converted to immutable MappingProxyType
        to maintain the frozen dataclass contract.
        """

        for field_info in fields(self):
            field_value = getattr(self, field_info.name)

            if field_value is None:
                continue

            # Convert dict to MappingProxyType
            if isinstance(field_value, dict):
                object.__setattr__(self, field_info.name, MappingProxyType(field_value))

    def convert_command_args(self) -> tuple[str, ...]:
        """Convert this option object to command-line arguments.

        Processes fields that contain Terraform option metadata and convert them
        to appropriate command-line arguments using their associated logic.

        Returns:
            A tuple of command-line arguments suitable for Terraform CLI.
        """

        result = self.command
        positional_args = tuple()
        sorted_fields = sorted(fields(self), key=lambda f: f.name)

        for _field in sorted_fields:
            tf_meta: FlagOption = _field.metadata.get(TF_OPTION_METANAME)
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
        raise NotImplementedError("command_type must be implemented by subclasses")

    def convert_option(
        self,
        target_class: Type[TFCommandOptions],
        **override_values,
    ) -> TFCommandOptions:
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
    """Represents options related to Terraform output formatting.

    This dataclass provides configuration options for formatting Terraform CLI output.
    Each field corresponds to a specific command-line flag used in Terraform.

    Attributes:
        json (bool | None): Controls whether the output is formatted as JSON.
            If True, Terraform output is formatted as JSON.
            If None, the option is omitted from CLI arguments.


        no_color (bool | None): Controls whether color codes are removed from Terraform output.
            If True, Terraform output will be rendered without ANSI color codes.
            If None, the option is omitted from CLI arguments.
    """

    json: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )

    no_color: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )


@dataclass(frozen=True)
class LockOptions:
    """Represents state locking options for Terraform commands.

    This dataclass extends `CommonOptions` to include configuration settings
    for state locking, which ensures safe concurrent access to Terraform state files.
    It is used by commands that either modify or read Terraform state.

    Attributes:
        lock (bool | None): Controls whether state locking is enabled.
            If True, Terraform will attempt to acquire a state lock before execution.
            If False, Terraform operations proceed without acquiring a state lock.
            If None, the option is omitted from CLI arguments.

        lock_timeout (int | None): Specifies the maximum duration (in seconds)
            to retry acquiring a state lock before failing.
            If set, Terraform will continue retrying until the timeout is reached.
            If None, the option is omitted from CLI arguments.
    """

    lock: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META},
    )

    lock_timeout: int | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=lambda n, v: (
                    (f"{n}={str(v)}s",) if v is not None else ()
                ),
            )
        },
    )


@dataclass(frozen=True)
class InputOptions:
    """Represents input handling options for Terraform commands.

    This dataclass provides configuration settings for controlling
    whether Terraform prompts for input during execution.

    Attributes:
        input (bool | None): Controls whether Terraform should request user input.
            If True, Terraform will prompt for missing variables and options.
            If False, Terraform will run in non-interactive mode without prompting.
            If None, the option is omitted from CLI arguments.
    """

    input: bool | None = field(
        init=False,
        default=False,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META},
    )


@dataclass(frozen=True)
class InitTaskOptions(OutputOptions, LockOptions, InputOptions, TFCommandOptions):
    """Represents options for the Terraform `init` command.

    This dataclass extends multiple option classes to provide configuration
    settings specific to the `terraform init` command. It includes options
    for backend initialization, state locking, input behavior, and output formatting.

    Attributes:
        backend (bool | None): Controls whether the backend should be initialized.
            If True, Terraform will initialize the configured backend.
            If False, backend initialization is skipped.
            If None, the option is omitted from CLI arguments.

        backend_config (str | MappingProxyType[str, str] | None): Specify the backend configuration.
            This can be provided as a file path or as key-value pairs for backend settings.
            If None, the option is omitted from CLI arguments.

        upgrade (bool | None): Controls whether Terraform should upgrade modules and plugins.
            If True, Terraform updates installed modules and providers to the latest versions.
            If False, existing versions are preserved.
            If None, the option is omitted from CLI arguments.
    """

    backend: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META},
    )

    backend_config: str | MappingProxyType[str, str] | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=backend_config_option,
            )
        },
    )

    upgrade: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )

    # NOTE Exclude because some versions are not supported
    json: bool | None = field(
        init=False,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )

    @property
    def command(self) -> tuple[str, ...]:
        return ("init",)


@dataclass(frozen=True)
class PlanApplyOptionBase(LockOptions, OutputOptions, InputOptions):
    """Represents options for the Terraform `plan` or `apply` command.

    This dataclass extends multiple option classes to provide configuration
    settings specific to the `terraform plan` or `terraform apply` command. It includes options
    for defining plan behavior, resource targeting, variable handling, and concurrency.

    Attributes:
        destroy (bool | None): Controls whether Terraform creates a plan to destroy all resources.
            If True, Terraform generates a plan that removes all managed infrastructures.
            If False, Terraform follows the standard planning process without destruction.
            If None, the option is omitted from CLI arguments.

        target (tuple[str, ...] | None): Limits planning to specific resources.
            Accepts one or more resource addresses to restrict changes to selected resources.
            If None, the option is omitted from CLI arguments.

        var (MappingProxyType[str, Any] | None): Specifies input variables for this run.
            Provides variables as key-value pairs to customize plan execution.
            If None, the option is omitted from CLI arguments.

        var_file (tuple[str, ...] | None): Loads variable files to provide predefined configurations.
            Accepts one or more file paths containing Terraform variable definitions.
            If None, the option is omitted from CLI arguments.

        parallelism (int | None): Limits concurrent operations during plan execution.
            Specify the maximum number of resources to process in parallel.
            If None, the option is omitted from CLI arguments.
    """

    destroy: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )

    target: tuple[str, ...] | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=list_str_option,
            )
        },
    )

    var: MappingProxyType[str, Any] | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=var_option,
            )
        },
    )

    var_file: tuple[str, ...] | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=list_str_option,
            )
        },
    )

    parallelism: int | None = field(
        init=True,
        default=None,
        metadata={
            TF_OPTION_METANAME: FlagOption(
                opt_name_func=underscore_to_hyphen,
                opt_values_func=lambda n, v: (
                    (f"{n}={str(v)}",) if v is not None else ()
                ),
            )
        },
    )


@dataclass(frozen=True)
class PlanTaskOptions(PlanApplyOptionBase, TFCommandOptions):
    """Represents options for the Terraform `plan` command.

    This dataclass extends multiple option classes to provide configuration
    settings specific to the `terraform plan` command. It includes options
    for defining plan behavior, resource targeting, variable handling, and concurrency.
    """

    @property
    def command(self) -> tuple[str, ...]:
        return ("plan",)


@dataclass(frozen=True)
class ApplyTaskOptions(PlanApplyOptionBase, TFCommandOptions):
    """Represents options for the Terraform `apply` command.

    This dataclass extends `PlanCommandOptions` to include configuration settings
    specific to the `terraform apply` command. It inherits planning-related options
    while adding settings for automatic execution control.

    Attributes:
        auto_approve (bool): Controls whether Terraform applies changes without user confirmation.
            If True, Terraform skips the interactive approval prompt and proceeds with execution.
            If False, Terraform requires explicit confirmation before applying the plan.
    """

    auto_approve: bool = field(
        init=False,
        default=True,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_FLAG_OPTION_META},
    )

    @property
    def command(self) -> tuple[str, ...]:
        return ("apply",)


@dataclass(frozen=True)
class OutputTaskOptions(OutputOptions, TFCommandOptions):
    """Represents options for the Terraform `output` command.

    This dataclass extends `OutputOptions` to provide configuration settings
    specific to `terraform output`. It allows customization of how output values
    are formatted and returned.
    """

    @property
    def command(self) -> tuple[str, ...]:
        return ("output",)


@dataclass(frozen=True)
class WorkspaceSelectTaskOptions(TFCommandOptions):
    """Represents options for the Terraform `workspace select` command."""

    workspace: str = field(
        init=True,
        metadata={TF_OPTION_METANAME: PositionalOption()},
    )

    or_create: bool | None = field(
        init=True,
        default=None,
        metadata={TF_OPTION_METANAME: UNDERSCORE_BOOL_OPTION_META},
    )

    @property
    def command(self) -> tuple[str, ...]:
        return "workspace", "select"
