import pytest

from twrapform.exception import TwrapformPreconditionError
from twrapform.options import InitTaskOptions
from twrapform.result import (
    PreExecutionFailure,
    TwrapformCommandTaskResult,
    TwrapformResult,
    TwrapformTaskError,
)


def test_twrapform_command_task_result_success():
    result = TwrapformCommandTaskResult(
        task_id="task_1",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="Success output",
        stderr="",
    )
    assert result.is_success() is True
    assert result.summary() == "[task_1] Completed with code 0"


def test_twrapform_command_task_result_error():
    result = TwrapformCommandTaskResult(
        task_id="task_2",
        task_option=InitTaskOptions(),
        return_code=1,
        stdout="",
        stderr="Error occurred",
    )
    assert result.is_success() is False
    with pytest.raises(TwrapformTaskError):
        result.raise_on_error()


def test_pre_execution_failure():
    original_error = ValueError("Invalid input")
    result = PreExecutionFailure(
        task_id="task_3", task_option=InitTaskOptions(), original_error=original_error
    )
    assert result.is_success() is False
    assert result.summary().startswith("[task_3] Failed before execution:")
    with pytest.raises(TwrapformPreconditionError):
        result.raise_on_error()


def test_twrapform_result():
    command_result1 = TwrapformCommandTaskResult(
        task_id="task_1",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="Success output",
        stderr="",
    )
    command_result2 = TwrapformCommandTaskResult(
        task_id="task_2",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="Success output",
        stderr="",
    )
    pre_failure = PreExecutionFailure(
        task_id="task_3",
        task_option=InitTaskOptions(),
        original_error=ValueError("Invalid input"),
    )
    twrap_result = TwrapformResult(
        task_results=tuple([command_result1, command_result2, pre_failure])
    )

    assert twrap_result.result_count == 3
    assert twrap_result.success_count == 2
    assert twrap_result.get_result("task_1") == command_result1
    assert twrap_result.get_result("task_2") == command_result2
    assert twrap_result.get_result("task_3") == pre_failure

    with pytest.raises(ValueError, match="No task result for task_id task_99"):
        twrap_result.get_result("task_99")

    with pytest.raises(TwrapformPreconditionError):
        twrap_result.raise_on_error()

    for success_task in twrap_result.get_success_tasks():
        assert isinstance(success_task, TwrapformCommandTaskResult)
        assert success_task.is_success() is True


def test_raise_on_error_multiple_results():
    success_result = TwrapformCommandTaskResult(
        task_id="task_4",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="All good",
        stderr="",
    )
    failure_result = TwrapformCommandTaskResult(
        task_id="task_5",
        task_option=InitTaskOptions(),
        return_code=2,
        stdout="Some output",
        stderr="Some error",
    )
    twrap_result = TwrapformResult(task_results=(success_result, failure_result))

    assert twrap_result.result_count == 2
    assert twrap_result.success_count == 1
    with pytest.raises(TwrapformTaskError):
        twrap_result.raise_on_error()

    for success_task in twrap_result.get_success_tasks():
        assert isinstance(success_task, TwrapformCommandTaskResult)
        assert success_task.is_success() is True
