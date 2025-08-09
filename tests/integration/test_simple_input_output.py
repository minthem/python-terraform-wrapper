import json
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from twrapform import Workflow
from twrapform.exception import (
    TwrapformError,
    TwrapformPreconditionError,
    TwrapformTaskError,
)
from twrapform.options import (
    ApplyTaskOptions,
    FrozenDict,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
    WorkspaceSelectTaskOptions,
)
from twrapform.result import CommandTaskResult


@pytest.fixture
def project_path():
    base_path = Path(__file__).parent / "terraform" / "simple_input_output"

    with TemporaryDirectory() as tmpdir:
        shutil.copytree(base_path, tmpdir, dirs_exist_ok=True)
        yield tmpdir


@pytest.mark.asyncio
async def test_execute_all_success(project_path):
    plan_option = PlanTaskOptions(
        var={
            "number": 100,
            "string": "hello",
            "boolean": True,
            "list": ["a", "b", "c"],
            "tuple": ("a", 123, False),
            "set": {"a", "b", "c"},
            "map": {"a": "123", "b": "456"},
            "object": {"field1": "123", "field2": "456"},
        }
    )
    twrapform = (
        Workflow(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(
            task_id="apply_1", task_option=plan_option.convert_option(ApplyTaskOptions)
        )
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.execute()

    assert results.result_count == 4

    try:
        results.raise_on_error()
    except (TwrapformTaskError, TwrapformPreconditionError) as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )


@pytest.mark.asyncio
async def test_execute_all_success_output_json(project_path):
    plan_option = PlanTaskOptions(
        var={
            "number": 100,
            "string": "hello",
            "boolean": True,
            "list": ["a", "b", "c"],
            "tuple": ("a", 123, False),
            "set": {"a", "b", "c"},
            "map": {"a": "123", "b": "456"},
            "object": {"field1": "123", "field2": "456"},
        },
        json=True,
    )
    twrapform = (
        Workflow(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(
            task_id="apply_1", task_option=plan_option.convert_option(ApplyTaskOptions)
        )
        .add_task(task_id="output_1", task_option=OutputTaskOptions(json=True))
    )

    results = await twrapform.execute(encoding_output=True)

    assert results.result_count == 4

    try:
        results.raise_on_error()
    except TwrapformError as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )

    try:
        for result in results.task_results:
            assert isinstance(result, CommandTaskResult)
            for line in result.stdout.split("\n"):
                json.dumps(line)
    except Exception as e:
        pytest.fail(e)


@pytest.mark.asyncio
async def test_execute_failed_and_resume(project_path):
    plan_option = PlanTaskOptions(
        var={
            "number": 100,
            "string": "hello",
            "boolean": True,
            "list": ["a", "b", "c"],
            "tuple": ("a", 123, False),
            "set": {"a", "b", "c"},
            "map": {"a": "123", "b": "456"},
            "object": {"field1": "123", "field2": "456"},
        }
    )
    twrapform = (
        Workflow(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(task_id="apply_1", task_option=ApplyTaskOptions())
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.execute()

    assert results.result_count == 3

    with pytest.raises(TwrapformError):
        results.raise_on_error()

    try:
        results.raise_on_error()
    except TwrapformError as e:
        fail_task = e.task_id

        new_twrapform = twrapform.change_task_option(
            fail_task, plan_option.convert_option(ApplyTaskOptions)
        )

        result_aft = await new_twrapform.execute(start_task_id=fail_task)

        assert result_aft.result_count == 2

        try:
            result_aft.raise_on_error()
        except TwrapformError as e:
            pytest.fail(e)


@pytest.mark.asyncio
async def test_execute_switch_ws(project_path):
    vars = {
        "number": 100,
        "string": "hello",
        "boolean": True,
        "list": ["a", "b", "c"],
        "tuple": ("a", 123, False),
        "set": {"a", "b", "c"},
        "map": {"a": "123", "b": "456"},
        "object": {"field1": "123", "field2": "456"},
    }
    twrapform = (
        Workflow(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(
            task_id="switch_workspace_1",
            task_option=WorkspaceSelectTaskOptions(workspace="test", or_create=True),
        )
        .add_task(task_id="plan_1", task_option=PlanTaskOptions(var=vars))
        .add_task(task_id="apply_1", task_option=ApplyTaskOptions(var=vars))
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.execute()

    assert results.result_count == 5

    try:
        results.raise_on_error()
    except TwrapformError as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )


@pytest.mark.asyncio
async def test_execute_output_json_hook(project_path, caplog):
    plan_vars = {
        "number": 100,
        "string": "hello",
        "boolean": True,
        "list": ["a", "b", "c"],
        "tuple": ("a", 123, False),
        "set": {"a", "b", "c"},
        "map": {"a": "123", "b": "456"},
        "object": {"field1": "123", "field2": "456"},
    }

    def _json_serializer(obj):
        if isinstance(obj, FrozenDict):
            return obj.export()
        elif isinstance(obj, frozenset | set | tuple):
            return list(obj)

        return obj

    def output_json_hook(workdir, task_option):
        with open(Path(workdir) / "twrapform.auto.tfvars.json", "w") as tfvars_json:
            if isinstance(task_option, (PlanTaskOptions, ApplyTaskOptions)):
                json.dump(
                    task_option.var.export(),
                    tfvars_json,
                    indent=2,
                    default=_json_serializer,
                )

    logger = logging.getLogger("twrapform_test")

    def logging_result(workdir, task_option, result):
        logger.info(
            f"directory: {workdir}, command: {task_option.command}, result: {result.is_success()}"
        )

    twrapform = (
        Workflow(work_dir=project_path)
        .add_task(
            task_id="init_1", task_option=InitTaskOptions(), post_hooks=[logging_result]
        )
        .add_task(
            task_id="plan_1",
            task_option=PlanTaskOptions(var=plan_vars),
            pre_hooks=[output_json_hook],
            post_hooks=[logging_result],
        )
        .add_task(
            task_id="apply_1",
            task_option=ApplyTaskOptions(),
            post_hooks=[logging_result],
        )
        .add_task(
            task_id="output_1",
            task_option=OutputTaskOptions(),
            post_hooks=[logging_result],
        )
    )

    with caplog.at_level(logging.INFO, logger="twrapform_test"):
        results = await twrapform.execute(encoding_output=True)

    assert results.result_count == 4

    try:
        results.raise_on_error()
    except TwrapformError as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )

    # check call pre hooks
    assert (Path(project_path) / "twrapform.auto.tfvars.json").exists()

    # check call post hooks
    info_logs = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(info_logs) == 4
