import textwrap

import pytest

from twrapform.exception import (
    TwrapformError,
    TwrapformPreconditionError,
    TwrapformTaskError,
)


@pytest.mark.unittest
def test_twrapform_error():
    error = TwrapformError(message="Test error")
    assert error.message == "Test error"


@pytest.mark.unittest
def test_twrapform_precondition_error():
    exc = KeyError("var")
    error = TwrapformPreconditionError(task_id="456", exc=exc)
    assert error.task_id == "456"
    assert error.original_exception == exc
    assert (
        error.message
        == "Twrapform precondition error, original error = KeyError('var')"
    )


@pytest.mark.unittest
def test_twrapform_task_error():
    error = TwrapformTaskError(
        task_id="789",
        return_code=1,
        stdout=textwrap.dedent(
            """
            Success output
            Task Done
            """[
                1:-1
            ]
        ),
        stderr="Error output",
    )
    assert error.task_id == "789"
    assert error.return_code == 1
    assert error.stdout == "Success output\nTask Done\n"
    assert error.stderr == "Error output"
    assert error.message == (
        "Terraform command failed with return code 1, stdout: Success output\nTask Done\n, stderr: Error output"
    )
