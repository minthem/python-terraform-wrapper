import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from twrapform import Task, Workflow, WorkflowManager
from twrapform.options import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
)


@pytest.fixture()
def project_path_base():
    base_path = Path(__file__).parent / "terraform" / "workflow_manager"

    with TemporaryDirectory() as tmpdir:
        shutil.copytree(base_path, tmpdir, dirs_exist_ok=True)
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def base_tasks():
    var = {
        "number": 100,
        "string": "hello",
        "boolean": True,
        "list": ["a", "b", "c"],
        "tuple": ("a", 123, False),
        "set": {"a", "b", "c"},
        "map": {"a": "123", "b": "456"},
        "object": {"field1": "123", "field2": "456"},
    }

    return (
        Task(task_id="init_1", option=InitTaskOptions()),
        Task(task_id="plan_1", option=PlanTaskOptions(var=var)),
        Task(task_id="apply_1", option=ApplyTaskOptions(var=var)),
        Task(task_id="output_1", option=OutputTaskOptions()),
    )


@pytest.fixture(scope="session")
def base_tasks_fail():
    var = {
        "number": 100,
        "object": {"field1": "123", "field2": "456"},
    }

    return (
        Task(task_id="init_1", option=InitTaskOptions()),
        Task(task_id="plan_1", option=PlanTaskOptions(var=var)),
        Task(task_id="apply_1", option=ApplyTaskOptions(var=var)),
        Task(task_id="output_1", option=OutputTaskOptions()),
    )


@pytest.fixture()
def workflow_1_1(project_path_base, base_tasks):
    return Workflow(work_dir=project_path_base / "stage1" / "task1", tasks=base_tasks)


@pytest.fixture()
def workflow_1_2(project_path_base, base_tasks):
    return Workflow(work_dir=project_path_base / "stage1" / "task2", tasks=base_tasks)


@pytest.fixture()
def workflow_2_1(project_path_base, base_tasks):
    return Workflow(work_dir=project_path_base / "stage2" / "task1", tasks=base_tasks)


@pytest.fixture()
def workflow_2_1_fail(project_path_base, base_tasks_fail):
    return Workflow(
        work_dir=project_path_base / "stage2" / "task1", tasks=base_tasks_fail
    )


@pytest.fixture()
def workflow_2_2(project_path_base, base_tasks):
    return Workflow(work_dir=project_path_base / "stage2" / "task2", tasks=base_tasks)


@pytest.fixture()
def workflow_3_1(project_path_base, base_tasks):
    return Workflow(work_dir=project_path_base / "stage3" / "task1", tasks=base_tasks)


@pytest.mark.asyncio
async def test_all_success(
    workflow_1_1, workflow_1_2, workflow_2_1, workflow_2_2, workflow_3_1
):
    manager = (
        WorkflowManager()
        .add_workflows(workflow_1_1, workflow_1_2)
        .add_workflows(workflow_2_1, workflow_2_2)
        .add_workflows(workflow_3_1)
    )

    result = await manager.execute()

    assert len(result.group_results) == 3
    assert result.success_count == 3
    assert result.is_all_success() == True


@pytest.mark.asyncio
async def test_fail_stop_workflow(
    workflow_1_1, workflow_1_2, workflow_2_1_fail, workflow_2_2, workflow_3_1
):
    manager = (
        WorkflowManager()
        .add_workflows(workflow_1_1, workflow_1_2)
        .add_workflows(workflow_2_1_fail, workflow_2_2)
        .add_workflows(workflow_3_1)
    )

    result = await manager.execute()

    assert len(result.group_results) == 2
    assert result.success_count == 1
    assert result.is_all_success() == False


@pytest.mark.asyncio
async def test_fail_dont_stop_workflow(
    workflow_1_1, workflow_1_2, workflow_2_1_fail, workflow_2_2, workflow_3_1
):
    manager = (
        WorkflowManager()
        .add_workflows(workflow_1_1, workflow_1_2)
        .add_workflows(workflow_2_1_fail, workflow_2_2)
        .add_workflows(workflow_3_1)
    )

    result = await manager.execute(stop_on_error=False)

    assert len(result.group_results) == 3
    assert result.success_count == 2
    assert result.is_all_success() == False
