import pytest

from twrapform.exception import (
    TwrapformError,
    TwrapformPreconditionError,
    WorkflowManagerExecutionError,
)
from twrapform.options import InitTaskOptions
from twrapform.result import (
    CommandTaskResult,
    PreExecutionFailure,
    TwrapformTaskError,
    WorkflowGroupResult,
    WorkflowManagerResult,
    WorkflowResult,
)


@pytest.fixture()
def success_result_1():
    return CommandTaskResult(
        task_id="task_1",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="Success output",
        stderr="",
    )


@pytest.fixture()
def success_result_2():
    return CommandTaskResult(
        task_id="task_2",
        task_option=InitTaskOptions(),
        return_code=0,
        stdout="Success output",
        stderr="",
    )


@pytest.fixture()
def error_result_1():
    return CommandTaskResult(
        task_id="task_3",
        task_option=InitTaskOptions(),
        return_code=1,
        stdout="",
        stderr="Error output",
    )


@pytest.fixture()
def error_result_2():
    return CommandTaskResult(
        task_id="task_4",
        task_option=InitTaskOptions(),
        return_code=2,
        stdout="Some output",
        stderr="Some error",
    )


@pytest.fixture()
def pre_execution_failure():
    return PreExecutionFailure(
        task_id="task_5",
        task_option=InitTaskOptions(),
        original_error=ValueError("Invalid input"),
    )


@pytest.fixture()
def success_workflow(success_result_1, success_result_2):
    return WorkflowResult(
        task_results=tuple([success_result_1, success_result_2]),
        workflow_id="w1",
    )


@pytest.fixture()
def success_workflows(success_result_1, success_result_2):
    ret = []

    for i in range(3):
        ret.append(
            WorkflowResult(
                task_results=tuple([success_result_1, success_result_2]),
                workflow_id=f"w{i + 1}",
            )
        )

    return tuple(ret)


@pytest.fixture()
def success_failed_tasks(success_result_1, success_result_2, error_result_1):
    s1 = WorkflowResult(
        task_results=tuple([success_result_1, success_result_2]),
        workflow_id="w1",
    )

    s2 = WorkflowResult(
        task_results=tuple([success_result_1, error_result_1]),
        workflow_id="w2",
    )

    return s1, s2


@pytest.fixture()
def success_group(success_workflow):
    return WorkflowGroupResult(group_id="g1", workflow_results=(success_workflow,))


@pytest.fixture()
def success_groups(success_group, success_workflow, success_workflows):
    return success_group, WorkflowGroupResult(
        group_id="g2", workflow_results=success_workflows
    )


@pytest.fixture()
def failed_group(success_failed_tasks):
    return WorkflowGroupResult(group_id="g3", workflow_results=success_failed_tasks)


class TestTwrapformWorkflowResult:
    def test_twrapform_command_task_result_success(self, success_result_1):
        assert success_result_1.is_success() is True
        assert (
            success_result_1.summary()
            == f"[{success_result_1.task_id}] Completed with code 0"
        )

    def test_twrapform_command_task_result_error(self, error_result_1):
        assert error_result_1.is_success() is False
        with pytest.raises(TwrapformTaskError):
            error_result_1.raise_on_error()

    def test_pre_execution_failure(self, pre_execution_failure):
        assert pre_execution_failure.is_success() is False
        assert pre_execution_failure.summary().startswith(
            f"[{pre_execution_failure.task_id}] Failed before execution:"
        )
        with pytest.raises(TwrapformPreconditionError):
            pre_execution_failure.raise_on_error()

    def test_twrapform_result(self, success_result_1, success_result_2):
        twrap_result = WorkflowResult(
            task_results=tuple([success_result_1, success_result_2]),
            workflow_id="w1",
        )

        assert twrap_result.result_count == 2
        assert twrap_result.success_count == 2
        assert twrap_result.get_result("task_1") == success_result_1
        assert twrap_result.get_result("task_2") == success_result_2
        assert twrap_result.is_all_success() == True

        for success_task in twrap_result.get_success_tasks():
            assert isinstance(success_task, CommandTaskResult)
            assert success_task.is_success() is True

    def test_twrapform_result_exist_failed(
        self, success_result_1, success_result_2, pre_execution_failure
    ):
        twrap_result = WorkflowResult(
            task_results=tuple(
                [
                    success_result_1,
                    success_result_2,
                    pre_execution_failure,
                ]
            ),
            workflow_id="w1",
        )

        assert twrap_result.result_count == 3
        assert twrap_result.success_count == 2
        assert twrap_result.get_result(success_result_1.task_id) == success_result_1
        assert twrap_result.get_result(success_result_2.task_id) == success_result_2
        assert (
            twrap_result.get_result(pre_execution_failure.task_id)
            == pre_execution_failure
        )
        assert twrap_result.is_all_success() == False

        with pytest.raises(ValueError, match="No task result for task_id task_99"):
            twrap_result.get_result("task_99")

        with pytest.raises(TwrapformPreconditionError):
            twrap_result.raise_on_error()

        for success_task in twrap_result.get_success_tasks():
            assert isinstance(success_task, CommandTaskResult)
            assert success_task.is_success() is True

    def test_raise_on_error_multiple_results(self, success_result_1, error_result_2):
        twrap_result = WorkflowResult(
            task_results=(success_result_1, error_result_2),
            workflow_id="w1",
        )

        assert twrap_result.result_count == 2
        assert twrap_result.success_count == 1
        assert twrap_result.is_all_success() == False
        with pytest.raises(TwrapformTaskError):
            twrap_result.raise_on_error()

        for success_task in twrap_result.get_success_tasks():
            assert isinstance(success_task, CommandTaskResult)
            assert success_task.is_success() is True


class TestWorkflowGroupResult:

    def test_workflow_group_result(self, success_workflow):
        group = WorkflowGroupResult(
            group_id="group1",
            workflow_results=tuple([success_workflow]),
        )

        try:
            group.raise_on_error()
        except TwrapformError:
            pytest.fail("raise_on_error() should not raise any exception")

        assert group.success_workflow_count == 1
        assert group.is_all_success() == True
        assert group.get_workflow_result("w1") == success_workflow

        with pytest.raises(ValueError, match="No workflow result for workflow_id w99"):
            group.get_workflow_result("w99")

    def test_workflow_group_result_multiple_success(self, success_workflows):
        group = WorkflowGroupResult(
            group_id="group1",
            workflow_results=success_workflows,
        )

        try:
            group.raise_on_error()
        except TwrapformError:
            pytest.fail("raise_on_error() should not raise any exception")

        assert group.success_workflow_count == 3
        assert group.is_all_success() == True
        assert group.get_workflow_result("w1") == group.workflow_results[0]
        assert group.get_workflow_result("w2") == group.workflow_results[1]
        assert group.get_workflow_result("w3") == group.workflow_results[2]

        with pytest.raises(ValueError, match="No workflow result for workflow_id w99"):
            group.get_workflow_result("w99")

    def test_workflow_group_result_multiple_failed(self, success_failed_tasks):
        group = WorkflowGroupResult(
            group_id="group1",
            workflow_results=success_failed_tasks,
        )

        with pytest.raises(TwrapformError):
            group.raise_on_error()

        assert group.success_workflow_count == 1
        assert group.is_all_success() == False
        assert group.get_workflow_result("w1") == success_failed_tasks[0]
        assert group.get_workflow_result("w2") == success_failed_tasks[1]


class TestWorkflowManagerResult:
    def test_workflow_manager_result(self, success_group):
        manager = WorkflowManagerResult(group_results=(success_group,))

        try:
            manager.raise_on_error()
        except TwrapformError:
            pytest.fail("raise_on_error() should not raise any exception")

        assert manager.success_count == 1
        assert manager.get_group_result("g1") == success_group

    def test_workflow_manager_multiple_result(self, success_groups):
        manager = WorkflowManagerResult(group_results=success_groups)

        try:
            manager.raise_on_error()
        except TwrapformError:
            pytest.fail("raise_on_error() should not raise any exception")

        assert manager.success_count == 2
        assert manager.get_group_result("g1") == success_groups[0]
        assert manager.get_group_result("g2") == success_groups[1]

    def test_workflow_manager_failed_result(self, success_groups, failed_group):
        manager = WorkflowManagerResult(group_results=(*success_groups, failed_group))

        with pytest.raises(WorkflowManagerExecutionError):
            manager.raise_on_error()

        assert manager.success_count == 2
        assert manager.get_group_result("g1") == success_groups[0]
        assert manager.get_group_result("g2") == success_groups[1]
        assert manager.get_group_result("g3") == failed_group

    def test_workflow_manager_not_exist_group(self, success_groups):
        manager = WorkflowManagerResult(group_results=success_groups)

        with pytest.raises(ValueError, match="No group result for group_id g99"):
            manager.get_group_result("g99")
