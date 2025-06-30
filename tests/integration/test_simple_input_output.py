import json
import shutil
from pathlib import Path

import pytest

from twrapform import Twrapform
from twrapform.exception import TwrapformError
from twrapform.result import TwrapformCommandTaskResult
from twrapform.task import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
    WorkspaceSelectTaskOptions,
)


@pytest.fixture
def project_path():
    path = Path(__file__).parent / "terraform" / "simple_input_output"
    shutil.rmtree(path / ".terraform", ignore_errors=True)
    shutil.rmtree(path / "terraform.tfstate.d", ignore_errors=True)

    yield Path(__file__).parent / "terraform" / "simple_input_output"

    shutil.rmtree(path / ".terraform", ignore_errors=True)
    shutil.rmtree(path / "terraform.tfstate.d", ignore_errors=True)


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
        Twrapform(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(
            task_id="apply_1", task_option=plan_option.convert_option(ApplyTaskOptions)
        )
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.run_await()

    assert len(results.task_results) == 4

    try:
        results.raise_on_error()
    except TwrapformError as e:
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
        Twrapform(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions(json=True))
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(
            task_id="apply_1", task_option=plan_option.convert_option(ApplyTaskOptions)
        )
        .add_task(task_id="output_1", task_option=OutputTaskOptions(json=True))
    )

    results = await twrapform.run_await()

    assert results.result_count == 4

    try:
        results.raise_on_error()
    except TwrapformError as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )

    try:
        for result in results.task_results:
            assert isinstance(result, TwrapformCommandTaskResult)
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
        Twrapform(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(task_id="plan_1", task_option=plan_option)
        .add_task(task_id="apply_1", task_option=ApplyTaskOptions())
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.run_await()

    assert len(results.task_results) == 3

    with pytest.raises(TwrapformError):
        results.raise_on_error()

    try:
        results.raise_on_error()
    except TwrapformError as e:
        fail_task = e.task_id

        new_twrapform = twrapform.change_task_option(
            fail_task, plan_option.convert_option(ApplyTaskOptions)
        )

        result_aft = await new_twrapform.run_await(start_task_id=fail_task)

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
        Twrapform(work_dir=project_path)
        .add_task(task_id="init_1", task_option=InitTaskOptions())
        .add_task(
            task_id="switch_workspace_1",
            task_option=WorkspaceSelectTaskOptions(workspace="test", or_create=True),
        )
        .add_task(task_id="plan_1", task_option=PlanTaskOptions(var=vars))
        .add_task(task_id="apply_1", task_option=ApplyTaskOptions(var=vars))
        .add_task(task_id="output_1", task_option=OutputTaskOptions())
    )

    results = await twrapform.run_await()

    assert results.result_count == 5

    try:
        results.raise_on_error()
    except TwrapformError as e:
        pytest.fail(
            f"pytest failed: {e.message}\n {results.get_result(e.task_id).task_option.convert_command_args()}"
        )
