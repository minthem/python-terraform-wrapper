from unittest.mock import AsyncMock, patch

import pytest

from twrapform import Workflow, WorkflowManager
from twrapform.options import InitTaskOptions
from twrapform.workflow import WorkflowGroup


@pytest.fixture
def workflow_manager_1() -> WorkflowManager:
    return WorkflowManager().add_workflows(
        Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
        group_id="group1",
    )


@pytest.fixture
def workflow_manager_2(workflow_manager_1) -> WorkflowManager:
    return workflow_manager_1.add_workflows(
        Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
        group_id="group2",
    )


class TestWorkflowManager:
    def test_add_workflows_group1(self):
        manager = WorkflowManager()
        manager = manager.add_workflows(
            Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp3").add_task(InitTaskOptions()),
            group_id="group1",
        )

        assert len(manager.groups) == 1
        assert manager.group_ids == ("group1",)

    def test_add_workflows_group1(self):
        manager = WorkflowManager()
        manager = manager.add_workflows(
            Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp3").add_task(InitTaskOptions()),
            group_id="group1",
        ).add_workflows(
            Workflow(work_dir="/tmp4").add_task(InitTaskOptions()),
            group_id="group2",
        )

        assert len(manager.groups) == 2
        assert manager.group_ids == ("group1", "group2")

    def test_duplicate_group_id(self, workflow_manager_1):

        with pytest.raises(ValueError, match="Group ID group1 already exists"):
            workflow_manager_1.add_workflows(
                Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
                group_id="group1",
            )

    def test_remove_group(self, workflow_manager_2):
        manager = workflow_manager_2.remove_group("group2")

        assert manager.group_ids == ("group1",)
        assert len(manager.groups) == 1

    def test_remove_group_not_exist_id(self, workflow_manager_2):
        with pytest.raises(ValueError, match="Group ID group99 does not exist"):
            workflow_manager_2.remove_group("group99")

    def test_clear_groups(self, workflow_manager_2):
        manager = workflow_manager_2.clear_groups()
        assert manager.group_ids == ()
        assert len(manager.groups) == 0

    def test_duplicate_group_id_post_init(self, workflow_manager_1):
        g1 = WorkflowGroup(
            group_id="group1",
            workflows=(),
            max_concurrency=2,
        )
        with pytest.raises(ValueError, match="Group ID must be unique"):
            WorkflowManager(groups=(g1, g1))

    def test_group_id_not_specific(self, workflow_manager_1):
        g1 = WorkflowGroup(
            group_id=None,
            workflows=(),
            max_concurrency=2,
        )
        with pytest.raises(ValueError, match="Group ID must be specified"):
            WorkflowManager(groups=(g1,))

    def test_add_workflows_less_than_max_concurrency(self, workflow_manager_1):
        with pytest.raises(ValueError, match="max_concurrency must be greater than 0"):
            manager = workflow_manager_1.add_workflows(
                Workflow(work_dir="/tmp4").add_task(InitTaskOptions()),
                max_concurrency=0,
            )


class TestWorkflowManagerExecute:
    @pytest.fixture()
    def success_parallel(self):
        return WorkflowManager().add_workflows(
            Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
            Workflow(work_dir="/tmp3").add_task(InitTaskOptions()),
            group_id="group1",
        )

    @pytest.fixture()
    def success_serial(self):
        return (
            WorkflowManager()
            .add_workflows(
                Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
                group_id="group1",
            )
            .add_workflows(
                Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
                group_id="group2",
            )
            .add_workflows(
                Workflow(work_dir="/tmp3").add_task(InitTaskOptions()),
                group_id="group3",
            )
            .add_workflows(
                Workflow(work_dir="/tmp4").add_task(InitTaskOptions()),
                group_id="group4",
            )
        )

    @pytest.fixture()
    def success_mix(self):
        return (
            WorkflowManager()
            .add_workflows(
                Workflow(work_dir="/tmp1").add_task(InitTaskOptions()),
                Workflow(work_dir="/tmp2").add_task(InitTaskOptions()),
                group_id="group1",
            )
            .add_workflows(
                Workflow(work_dir="/tmp3").add_task(InitTaskOptions()),
                Workflow(work_dir="/tmp4").add_task(InitTaskOptions()),
                group_id="group2",
            )
            .add_workflows(
                Workflow(work_dir="/tmp5").add_task(InitTaskOptions()),
                group_id="group3",
            )
        )

    @pytest.fixture()
    def success_proc(self):
        proc = AsyncMock()
        proc.communicate.return_value = (
            "success",
            "",
        )
        proc.wait.return_value = 0
        return proc

    @pytest.fixture()
    def fail_proc(self):
        proc = AsyncMock()
        proc.communicate.return_value = (
            "",
            "fail",
        )
        proc.wait.return_value = 1
        return proc

    @pytest.mark.asyncio
    async def test_execute_all_success_parallel(self, success_parallel, success_proc):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = success_proc

            result = await success_parallel.execute()

            assert mock_exec.call_count == 3
            assert len(result.group_results) == 1

    @pytest.mark.asyncio
    async def test_execute_all_success_serial(self, success_serial, success_proc):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = success_proc

            result = await success_serial.execute()

            assert mock_exec.call_count == 4
            assert len(result.group_results) == 4

    @pytest.mark.asyncio
    async def test_execute_all_success_mix(self, success_mix, success_proc):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = success_proc

            result = await success_mix.execute()

            assert mock_exec.call_count == 5
            assert len(result.group_results) == 3

    @pytest.mark.asyncio
    async def test_execute_fail_task(self, success_serial, success_proc, fail_proc):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [
                success_proc,
                success_proc,
                fail_proc,
                success_proc,
            ]

            result = await success_serial.execute()

            assert mock_exec.call_count == 3
            assert len(result.group_results) == 3
            assert result.success_count == 2

    @pytest.mark.asyncio
    async def test_execute_fail_task_no_stop(
        self, success_serial, success_proc, fail_proc
    ):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [
                success_proc,
                success_proc,
                fail_proc,
                success_proc,
            ]

            result = await success_serial.execute(stop_on_error=False)

            assert mock_exec.call_count == 4
            assert len(result.group_results) == 4
            assert result.success_count == 3
