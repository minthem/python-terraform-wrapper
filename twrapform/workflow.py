from __future__ import annotations

import asyncio
import locale
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from .common import (
    GroupID,
    Task,
    TaskID,
    WorkflowID,
    gen_group_id,
    gen_sequential_id,
    gen_workflow_id,
)
from .logging import get_logger
from .options import SupportedTerraformTask
from .result import (
    CommandTaskResult,
    PreExecutionFailure,
    WorkflowGroupResult,
    WorkflowManagerResult,
    WorkflowResult,
)


@dataclass(frozen=True)
class Workflow:
    """Twrapform configuration object."""

    work_dir: os.PathLike[str] | str
    terraform_path: os.PathLike[str] | str = "terraform"
    tasks: tuple[Task, ...] = field(default_factory=tuple)
    workflow_id: WorkflowID = field(default_factory=gen_workflow_id)

    def __post_init__(self):
        task_ids = set(self.task_ids)

        if None in task_ids:
            raise ValueError("Task ID must be specified")

        if len(task_ids) != len(self.tasks):
            raise ValueError("Task ID must be unique")

    @property
    def task_ids(self) -> tuple[TaskID, ...]:
        return tuple(task.task_id for task in self.tasks)

    def exist_task(self, task_id: TaskID) -> bool:
        return task_id in self.task_ids

    def get_task(self, task_id: TaskID) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        else:
            raise ValueError(f"Task ID {task_id} does not exist")

    def add_task(
        self, task_option: SupportedTerraformTask, task_id: TaskID | None = None
    ) -> Workflow:
        """Add a task to the Twrapform object with the specified options and ID.

        Args:
            task_option: The Terraform task configuration to be added.
            task_id: Optional unique identifier for the task. If not provided,
                    an ID will be generated using the task command and a sequential number.

        Returns:
            Workflow: A new Workflow instance with the added task.

        Raises:
            ValueError: If the provided task_id already exists in the workflow.
        """

        task_ids = self.task_ids
        if task_id is None:
            task_id = "_".join([*task_option.command, gen_sequential_id()])

        else:
            if task_id in task_ids:
                raise ValueError(f"Task ID {task_id} already exists")

        return replace(
            self,
            tasks=tuple([*self.tasks, Task(task_id=task_id, option=task_option)]),
        )

    def change_task_option(self, task_id: TaskID, new_option: SupportedTerraformTask):
        """Change the option of a task."""
        task_index = self._get_task_index(task_id)
        new_tasks = (
            *self.tasks[:task_index],
            Task(task_id=task_id, option=new_option),
            *self.tasks[task_index + 1 :],
        )

        return replace(
            self,
            tasks=tuple(new_tasks),
        )

    def remove_task(self, task_id: TaskID) -> Workflow:
        """Remove a task from the Twrapform object."""

        if not self.exist_task(task_id):
            raise ValueError(f"Task ID {task_id} does not exist")

        task_id_index = self._get_task_index(task_id)

        new_tasks = tuple(
            [*self.tasks[:task_id_index], *self.tasks[task_id_index + 1 :]]
        )

        return replace(self, tasks=new_tasks)

    def clear_tasks(self) -> Workflow:
        """Remove all tasks from the Twrapform object."""
        return replace(self, tasks=tuple())

    def _get_task_index(self, task_id: TaskID) -> int:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return index
        else:
            raise ValueError(f"Task ID {task_id} does not exist")

    async def execute(
        self,
        *,
        start_task_id: TaskID | None = None,
        encoding_output: bool | str = False,
    ) -> WorkflowResult:
        """Run all tasks asynchronously."""

        env_vars = os.environ.copy()

        if start_task_id is not None:
            start_index = self._get_task_index(start_task_id)
        else:
            start_index = 0

        encoding = _get_encoding_output(encoding_output)

        results = await _execute_terraform_tasks(
            work_dir=self.work_dir,
            terraform_path=self.terraform_path,
            tasks=self.tasks[start_index:],
            env_vars=env_vars,
            output_encoding=encoding,
        )

        return WorkflowResult(workflow_id=self.workflow_id, task_results=results)


class WorkflowGroup(NamedTuple):
    group_id: GroupID
    workflows: tuple[Workflow, ...]
    max_concurrency: int


@dataclass(frozen=True)
class WorkflowManager:
    """Twrapform configuration object."""

    groups: tuple[WorkflowGroup, ...] = field(default_factory=tuple)

    def __post_init__(self):
        group_ids = set(self.group_ids)

        if None in group_ids:
            raise ValueError("Group ID must be specified")

        if len(group_ids) != len(self.group_ids):
            raise ValueError("Group ID must be unique")

    @property
    def group_ids(self) -> tuple[GroupID, ...]:
        return tuple(workflow.group_id for workflow in self.groups)

    def add_workflows(
        self,
        *workflows: Workflow,
        group_id: GroupID | None = None,
        max_concurrency: int | None = None,
    ) -> WorkflowManager:
        """Add workflows to the Twrapform object and create a new workflow group.

        Args:
            *workflows: Variable number of Workflow objects to add to the group.
            group_id: Optional unique identifier for the group. If not provided, a new ID will be generated.
            max_concurrency: Maximum number of workflows that can run concurrently in this group.
                           If not provided, defaults to the number of workflows in the group.
                           Must be greater than 0 if specified.

        Returns:
            WorkflowManager: A new WorkflowManager instance with the added workflow group.

        Raises:
            ValueError: If group_id already exists or max_concurrency is less than or equal to 0.
        """
        group_ids = self.group_ids
        if group_id is None:
            group_id = gen_group_id()
        else:
            if group_id in group_ids:
                raise ValueError(f"Group ID {group_id} already exists")

        if isinstance(workflows, Workflow):
            workflows = (workflows,)

        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")

        if max_concurrency is None:
            max_concurrency = len(workflows)

        workflow_group = WorkflowGroup(group_id, workflows, max_concurrency)

        return replace(
            self,
            groups=self.groups + (workflow_group,),
        )

    def exist_group(self, group_id: GroupID) -> bool:
        """Check if a workflow exists in the Twrapform object."""

        group_ids = self.group_ids
        return group_id in group_ids

    def _get_group_index(self, group_id: GroupID) -> int:
        for index, group in enumerate(self.groups):
            if group.group_id == group_id:
                return index
        else:
            raise ValueError(f"Group ID {group_id} does not exist")

    def remove_group(self, group_id: GroupID) -> WorkflowManager:
        """Remove a group from the Twrapform object."""
        if not self.exist_group(group_id):
            raise ValueError(f"Group ID {group_id} does not exist")

        group_id_index = self._get_group_index(group_id)

        new_groups = tuple(
            [*self.groups[:group_id_index], *self.groups[group_id_index + 1 :]]
        )

        return replace(self, groups=new_groups)

    def clear_groups(self) -> WorkflowManager:
        """Remove all groups from the Twrapform object."""
        return replace(self, groups=tuple())

    async def execute(
        self,
        *,
        start_group_id: GroupID | None = None,
        encoding_output: bool | str = False,
        stop_on_error: bool = True,
    ) -> WorkflowManagerResult:
        """Run all workflows asynchronously in each group with concurrency control.

        Args:
            start_group_id: Optional group ID to start execution from. If provided,
                          only groups from this ID onwards will be executed.
            encoding_output: Controls output encoding for command results. If True, uses system
                          preferred encoding. If string, uses specified encoding. If False,
                          leaves output as bytes.
            stop_on_error: If True, stops execution when any workflow in a group fails.
                         If False, continues to next group regardless of failures.

        Returns:
            WorkflowManagerResult: Object containing results of all executed workflow groups.
        """
        results: list[WorkflowGroupResult] = []

        if start_group_id is not None:
            start_index = self._get_group_index(start_group_id)
        else:
            start_index = 0

        encoding = _get_encoding_output(encoding_output)

        async def _execute_task_sem(workflow: Workflow, sem: asyncio.Semaphore):
            async with sem:
                return await workflow.execute(encoding_output=encoding)

        for group in self.groups[start_index:]:
            group_jobs = [
                _execute_task_sem(workflow, asyncio.Semaphore(group.max_concurrency))
                for workflow in group.workflows
            ]

            task_results = await asyncio.gather(*group_jobs)
            group_results = WorkflowGroupResult(
                group_id=group.group_id, workflow_results=tuple(task_results)
            )
            results.append(group_results)

            if stop_on_error and not group_results.is_all_success():
                break

        return WorkflowManagerResult(group_results=tuple(results))


async def _execute_terraform_tasks(
    work_dir: os.PathLike[str] | str,
    terraform_path: os.PathLike[str] | str,
    tasks: tuple[Task, ...],
    env_vars: dict[str, str] | None = None,
    output_encoding: str | None = None,
    logger: logging.Logger = get_logger(),
) -> tuple[CommandTaskResult | PreExecutionFailure, ...]:
    task_results = []

    if env_vars is None:
        env_vars = os.environ.copy()

    for task in tasks:
        try:
            if shutil.which(terraform_path) is None:
                raise FileNotFoundError(
                    f"Terraform executable not found: {terraform_path}"
                )
            cmd_args = (
                f"-chdir={work_dir}",
                *task.option.convert_command_args(),
            )
            proc = await asyncio.create_subprocess_exec(
                terraform_path,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars,
            )

            stdout, stderr = await proc.communicate()
            return_code = await proc.wait()

            if output_encoding is not None:
                try:
                    stdout = stdout.decode(output_encoding)
                    stderr = stderr.decode(output_encoding)
                except UnicodeDecodeError as e:
                    logger.warning("[%s] Failed encoding output: %s", task.task_id, e)

            task_results.append(
                CommandTaskResult(
                    task_id=task.task_id,
                    task_option=task.option,
                    return_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                )
            )

            if return_code != 0:
                break
        except Exception as e:
            error = PreExecutionFailure(
                task_id=task.task_id,
                original_error=e,
                task_option=task.option,
            )
            task_results.append(error)
            break

    return tuple(task_results)


def _get_encoding_output(encoding_output: bool | str | None) -> str | None:
    if isinstance(encoding_output, bool):
        if encoding_output:
            return locale.getpreferredencoding()

        return None

    return encoding_output
