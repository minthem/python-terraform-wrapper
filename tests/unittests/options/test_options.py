from dataclasses import fields
from types import MappingProxyType

import pytest

from twrapform.options import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
    WorkspaceSelectTaskOptions,
)


class TestInitTaskOptions:
    """Test cases for InitTaskOptions class."""

    @pytest.mark.parametrize(
        ("opt_args", "expected_result"),
        [
            (
                {},
                (
                    "init",
                    "-input=false",
                ),
            ),
            ({"backend": True}, ("init", "-backend=true", "-input=false")),
            (
                {"upgrade": True},
                (
                    "init",
                    "-input=false",
                    "-upgrade",
                ),
            ),
            (
                {"backend": True, "upgrade": True},
                ("init", "-backend=true", "-input=false", "-upgrade"),
            ),
            (
                {"backend_config": "/path/to/config"},
                ("init", "-backend-config=/path/to/config", "-input=false"),
            ),
            (
                {"backend_config": {"key": "value", "key2": "value2"}},
                (
                    "init",
                    "-backend-config='key=value'",
                    "-backend-config='key2=value2'",
                    "-input=false",
                ),
            ),
            (
                {"backend_config": {"key": "value"}},
                ("init", "-backend-config='key=value'", "-input=false"),
            ),
        ],
    )
    def test_convert_command_args_parametrized(self, opt_args, expected_result):
        """Test command line generation with various parameter combinations."""
        options = InitTaskOptions(**opt_args)
        result = options.convert_command_args()
        assert result == expected_result

    def test_backend_config_with_mapping_proxy(self):
        """Test backend_config with MappingProxyType."""
        backend_config = MappingProxyType({"key": "value"})
        options = InitTaskOptions(backend_config=backend_config)
        assert options.backend_config == backend_config

    def test_backend_config_raises_type_error(self):
        """Test backend_config raises TypeError."""
        with pytest.raises(TypeError):
            options = InitTaskOptions(backend_config=123).convert_command_args()

    @pytest.mark.parametrize(
        ("type_", "override_values"),
        [
            (InitTaskOptions, {"backend": False}),
            (PlanTaskOptions, {"target": ("resource.c",)}),
            (ApplyTaskOptions, {"target": ("resource.c",)}),
            (OutputTaskOptions, {"json": True}),
            (WorkspaceSelectTaskOptions, {"workspace": "foo"}),
        ],
    )
    def test_convert_option(self, type_, override_values):
        opt = InitTaskOptions(
            backend_config="/path/to/config",
            no_color=True,
            backend=True,
            upgrade=True,
            lock=False,
            lock_timeout=100,
        )

        new_opt = opt.convert_option(type_, **override_values)

        common_fields = {field.name for field in fields(opt)} & {
            field.name for field in fields(new_opt)
        }

        for field_name in common_fields:
            if field_name not in override_values:
                assert getattr(new_opt, field_name) == getattr(opt, field_name)

        for k, v in override_values.items():
            assert getattr(new_opt, k) == v


class TestPlanOptions:
    """Test cases for PlanOptions class."""

    @pytest.mark.parametrize(
        ("opt_args", "expected_result"),
        [
            (
                {},
                (
                    "plan",
                    "-input=false",
                ),
            ),
            (
                {"destroy": True},
                (
                    "plan",
                    "-destroy",
                    "-input=false",
                ),
            ),
            (
                {"parallelism": 10},
                (
                    "plan",
                    "-input=false",
                    "-parallelism=10",
                ),
            ),
            (
                {"target": ("resource.example",)},
                (
                    "plan",
                    "-input=false",
                    "-target=resource.example",
                ),
            ),
            (
                {"var_file": ("/path/to/vars.tf",)},
                (
                    "plan",
                    "-input=false",
                    "-var-file=/path/to/vars.tf",
                ),
            ),
            (
                {"destroy": True, "parallelism": 5},
                ("plan", "-destroy", "-input=false", "-parallelism=5"),
            ),
            (
                {"target": ("resource.a", "resource.b")},
                ("plan", "-input=false", "-target=resource.a", "-target=resource.b"),
            ),
            (
                {
                    "var": {
                        "boolean": True,
                        "integer": 5,
                        "float": 3.14,
                        "string": "string",
                        "list": [1, 2, 3],
                        "tuple": ("1", "2", "3"),
                        "set": {1, 2, 3},
                        "map": {"a": 1, "b": 2, "c": 3},
                    },
                },
                (
                    "plan",
                    "-input=false",
                    "-var",
                    "boolean=true",
                    "-var",
                    "float=3.14",
                    "-var",
                    "integer=5",
                    "-var",
                    "list=[1, 2, 3]",
                    "-var",
                    'map={"a": 1, "b": 2, "c": 3}',
                    "-var",
                    "set=[1, 2, 3]",
                    "-var",
                    "string=string",
                    "-var",
                    'tuple=["1", "2", "3"]',
                ),
            ),
        ],
    )
    def test_convert_command_args_parametrized(self, opt_args, expected_result):
        """Test command line generation with various parameter combinations."""
        options = PlanTaskOptions(**opt_args)
        result = options.convert_command_args()
        assert result == expected_result

    def test_convert_command_args_idempotent_with_complex_data(self):
        """Test that convert_command_args() is idempotent with complex data structures."""
        options = PlanTaskOptions(
            json=True,
            lock=False,
            destroy=True,
            parallelism=3,
            var=MappingProxyType(
                {"z_var": "last", "a_var": "first", "b_var": "second"}
            ),
            target=("resource.b", "resource.a"),
        )

        result1 = options.convert_command_args()
        result2 = options.convert_command_args()
        result3 = options.convert_command_args()

        assert result1 == result2 == result3
        assert isinstance(result1, tuple)

    def test_var_raises_type_error(self):
        """Test var raises TypeError."""
        with pytest.raises(TypeError):
            options = PlanTaskOptions(var={"foo": object()}).convert_command_args()

    @pytest.mark.parametrize(
        ("type_", "override_values"),
        [
            (InitTaskOptions, {"backend": False}),
            (PlanTaskOptions, {"target": ("resource.c",)}),
            (ApplyTaskOptions, {"target": ("resource.c",)}),
            (OutputTaskOptions, {"json": True}),
            (WorkspaceSelectTaskOptions, {"workspace": "foo"}),
        ],
    )
    def test_convert_option(self, type_, override_values):
        opt = PlanTaskOptions(
            no_color=True,
            lock=False,
            lock_timeout=100,
            var={
                "boolean": True,
                "integer": 5,
                "float": 3.14,
                "string": "string",
                "list": [1, 2, 3],
                "tuple": ("1", "2", "3"),
            },
            target=("resource.a", "resource.b"),
            var_file=("/path/to/vars.tf",),
            parallelism=5,
        )

        new_opt = opt.convert_option(type_, **override_values)

        common_fields = {field.name for field in fields(opt)} & {
            field.name for field in fields(new_opt)
        }

        for field_name in common_fields:
            if field_name not in override_values:
                assert getattr(new_opt, field_name) == getattr(opt, field_name)

        for k, v in override_values.items():
            assert getattr(new_opt, k) == v


class TestApplyTaskOptions:
    """Test cases for ApplyOptions class."""

    @pytest.mark.parametrize(
        ("opt_args", "expected_contains"),
        [
            (
                {},
                (
                    "apply",
                    "-auto-approve",
                    "-input=false",
                ),
            ),  # Default auto_approve=True
            (
                {"destroy": True},
                (
                    "apply",
                    "-auto-approve",
                    "-destroy",
                    "-input=false",
                ),
            ),
            (
                {"parallelism": 10},
                ("apply", "-auto-approve", "-input=false", "-parallelism=10"),
            ),
            (
                {"target": ("resource.example",)},
                ("apply", "-auto-approve", "-input=false", "-target=resource.example"),
            ),
            (
                {"destroy": True, "parallelism": 5},
                (
                    "apply",
                    "-auto-approve",
                    "-destroy",
                    "-input=false",
                    "-parallelism=5",
                ),
            ),
            (
                {
                    "var": {
                        "boolean": True,
                        "integer": 5,
                        "float": 3.14,
                        "string": "string",
                        "list": [1, 2, 3],
                        "tuple": ("1", "2", "3"),
                        "set": {1, 2, 3},
                        "map": {"a": 1, "b": 2, "c": 3},
                    },
                },
                (
                    "apply",
                    "-auto-approve",
                    "-input=false",
                    "-var",
                    "boolean=true",
                    "-var",
                    "float=3.14",
                    "-var",
                    "integer=5",
                    "-var",
                    "list=[1, 2, 3]",
                    "-var",
                    'map={"a": 1, "b": 2, "c": 3}',
                    "-var",
                    "set=[1, 2, 3]",
                    "-var",
                    "string=string",
                    "-var",
                    'tuple=["1", "2", "3"]',
                ),
            ),
        ],
    )
    def test_convert_command_args_parametrized(self, opt_args, expected_contains):
        """Test command line generation with various parameter combinations."""
        options = ApplyTaskOptions(**opt_args)
        result = options.convert_command_args()

        for expected_arg in expected_contains:
            assert expected_arg in result

    def test_convert_command_args_idempotent_with_complex_data(self):
        """Test that convert_command_args() is idempotent with complex data structures."""
        options = ApplyTaskOptions(
            json=True,
            lock=False,
            destroy=True,
            parallelism=3,
            var=MappingProxyType(
                {"z_var": "last", "a_var": "first", "b_var": "second"}
            ),
            target=("resource.b", "resource.a"),
        )

        result1 = options.convert_command_args()
        result2 = options.convert_command_args()
        result3 = options.convert_command_args()

        assert result1 == result2 == result3
        assert isinstance(result1, tuple)

    @pytest.mark.parametrize(
        ("type_", "override_values"),
        [
            (InitTaskOptions, {"backend": False}),
            (PlanTaskOptions, {"target": ("resource.c",)}),
            (ApplyTaskOptions, {"target": ("resource.c",)}),
            (OutputTaskOptions, {"json": True}),
            (WorkspaceSelectTaskOptions, {"workspace": "foo"}),
        ],
    )
    def test_convert_option(self, type_, override_values):
        opt = ApplyTaskOptions(
            no_color=True,
            json=True,
            lock=False,
            lock_timeout=100,
            var={
                "boolean": True,
                "integer": 5,
                "float": 3.14,
                "string": "string",
                "list": [1, 2, 3],
                "tuple": ("1", "2", "3"),
            },
            target=("resource.a", "resource.b"),
            var_file=("/path/to/vars.tf",),
            parallelism=5,
        )

        new_opt = opt.convert_option(type_, **override_values)

        common_fields = {field.name for field in fields(opt)} & {
            field.name for field in fields(new_opt) if field.init
        }

        for field_name in common_fields:
            if field_name not in override_values:
                assert getattr(new_opt, field_name) == getattr(opt, field_name)

        for k, v in override_values.items():
            assert getattr(new_opt, k) == v


class TestOutputOptions:
    """Test cases for OutputOptions class."""

    @pytest.mark.parametrize(
        ("opt_args", "expected_result"),
        [
            ({}, ("output",)),
            (
                {"json": True},
                (
                    "output",
                    "-json",
                ),
            ),
            (
                {"no_color": True},
                (
                    "output",
                    "-no-color",
                ),
            ),
            (
                {"json": False},
                ("output",),
            ),
            ({"json": True, "no_color": True}, ("output", "-json", "-no-color")),
        ],
    )
    def test_convert_command_args_parametrized(self, opt_args, expected_result):
        """Test command line generation with various parameter combinations."""
        options = OutputTaskOptions(**opt_args)
        result = options.convert_command_args()
        assert result == expected_result

    @pytest.mark.parametrize(
        ("type_", "override_values"),
        [
            (InitTaskOptions, {"backend": False}),
            (PlanTaskOptions, {"target": ("resource.c",)}),
            (ApplyTaskOptions, {"target": ("resource.c",)}),
            (OutputTaskOptions, {"json": True}),
            (WorkspaceSelectTaskOptions, {"workspace": "foo"}),
        ],
    )
    def test_convert_option(self, type_, override_values):
        opt = OutputTaskOptions(
            no_color=False,
            json=False,
        )

        new_opt = opt.convert_option(type_, **override_values)

        common_fields = {field.name for field in fields(opt)} & {
            field.name for field in fields(new_opt) if field.init
        }

        for field_name in common_fields:
            if field_name not in override_values:
                assert getattr(new_opt, field_name) == getattr(opt, field_name)

        for k, v in override_values.items():
            assert getattr(new_opt, k) == v


class TestWorkspaceSelectCommandOptions:
    """Test cases for WorkspaceSelectCommandOptions class."""

    @pytest.mark.parametrize(
        ("opt_args", "expected_result"),
        [
            (
                {"workspace": "test"},
                (
                    "workspace",
                    "select",
                    "test",
                ),
            ),
            (
                {"workspace": "test", "or_create": False},
                ("workspace", "select", "-or-create=false", "test"),
            ),
            (
                {
                    "workspace": "test",
                    "or_create": True,
                },
                ("workspace", "select", "-or-create=true", "test"),
            ),
        ],
    )
    def test_convert_command_args_parametrized(self, opt_args, expected_result):
        """Test command line generation with various parameter combinations."""
        options = WorkspaceSelectTaskOptions(**opt_args)
        result = options.convert_command_args()
        assert result == expected_result

    def test_convert_command_args_no_specific_workspace(self):
        """Test not specifying a workspace raises an error."""
        with pytest.raises(TypeError):
            WorkspaceSelectTaskOptions()

    @pytest.mark.parametrize(
        ("type_", "override_values"),
        [
            (InitTaskOptions, {"backend": False}),
            (PlanTaskOptions, {"target": ("resource.c",)}),
            (ApplyTaskOptions, {"target": ("resource.c",)}),
            (OutputTaskOptions, {"json": True}),
            (WorkspaceSelectTaskOptions, {"workspace": "foo"}),
        ],
    )
    def test_convert_option(self, type_, override_values):
        opt = WorkspaceSelectTaskOptions(
            workspace="test",
            or_create=True,
        )

        new_opt = opt.convert_option(type_, **override_values)

        common_fields = {field.name for field in fields(opt)} & {
            field.name for field in fields(new_opt)
        }

        for field_name in common_fields:
            if field_name not in override_values:
                assert getattr(new_opt, field_name) == getattr(opt, field_name)

        for k, v in override_values.items():
            assert getattr(new_opt, k) == v
