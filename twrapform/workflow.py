from __future__ import annotations

import asyncio
import copy
import locale
import logging
import os
import platform
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import hcl2

from ._common import (
    GroupID,
    TaskID,
    WorkflowID,
    gen_group_id,
    gen_sequential_id,
    gen_workflow_id,
)
from ._lock import AsyncResourceLockManager
from ._logging import get_logger
from .options import InitTaskOptions, TFCommandOptions
from .result import (
    CommandTaskResult,
    PreExecutionFailure,
    TaskResult,
    WorkflowGroupResult,
    WorkflowManagerResult,
    WorkflowResult,
)


def _get_terraform_provider_cache_dir(env) -> str | None:
    """Resolve Terraform provider plugin cache directory.

    Resolution order:
      1) TF_PLUGIN_CACHE_DIR in the provided env
      2) plugin_cache_dir in the Terraform CLI config (rc) file
         - Config file path is taken from TF_CLI_CONFIG_FILE if set,
           otherwise resolved by platform conventions:
             - Windows: %APPDATA%/terraform.rc
             - Others: ~/.terraformrc

    Args:
        env: Mapping of environment variables to inspect.

    Returns:
        The resolved directory path as a string, or None if it cannot be determined.
    """
    if "TF_PLUGIN_CACHE_DIR" in env:
        return env["TF_PLUGIN_CACHE_DIR"]

    if "TF_CLI_CONFIG_FILE" in env:
        conf_path = Path(env["TF_CLI_CONFIG_FILE"])
    else:
        os_name = platform.system()

        if os_name == "Windows" and "APPDATA" in env:
            conf_path = Path(env["APPDATA"]) / "terraform.rc"
        else:
            conf_path = Path.home() / ".terraformrc"

    if not conf_path.exists():
        return None

    with conf_path.open(mode="r") as f:
        terraform_rc = hcl2.load(f)

        if "plugin_cache_dir" in terraform_rc:
            return terraform_rc["plugin_cache_dir"]

    return None


_lock_manager_instance = AsyncResourceLockManager()
_provider_cache_dir = _get_terraform_provider_cache_dir(os.environ)

PreHook = Callable[[str | os.PathLike[str], TFCommandOptions], None]
PostHook = Callable[[str | os.PathLike[str], TFCommandOptions, TaskResult], None]


class Task(NamedTuple):
    """Execution unit representing a single Terraform command invocation.

    Attributes:
        task_id: Unique identifier for this task within a workflow.
        option: A TFCommandOptions describing the Terraform command and its arguments.
        pre_hooks: Hooks executed before the command. Each receives (work_dir, option_copy).
        post_hooks: Hooks executed after the command. Each receives (work_dir, option_copy, task_result_copy).
    """

    task_id: TaskID
    option: TFCommandOptions
    pre_hooks: tuple[PreHook, ...] = ()
    post_hooks: tuple[PostHook, ...] = ()


@dataclass(frozen=True)
class Workflow:
    """Twrapform configuration object.

    A Workflow encapsulates a working directory, a Terraform executable path,
    and an ordered list of tasks to be executed in sequence.
    """

    work_dir: os.PathLike[str] | str
    terraform_path: os.PathLike[str] | str = "terraform"
    tasks: tuple[Task, ...] = field(default_factory=tuple)
    workflow_id: WorkflowID = field(default_factory=gen_workflow_id)

    def __post_init__(self):
        """Validate task identifiers after initialization.

        Ensures that:
          - All tasks have a non-None task_id
          - task_ids are unique within this workflow

        Raises:
            ValueError: If a task_id is missing or duplicated.
        """
        task_ids = set(self.task_ids)

        if None in task_ids:
            raise ValueError("Task ID must be specified")

        if len(task_ids) != len(self.tasks):
            raise ValueError("Task ID must be unique")

    @property
    def task_ids(self) -> tuple[TaskID, ...]:
        """Tuple of task IDs in their execution order."""
        return tuple(task.task_id for task in self.tasks)

    def exist_task(self, task_id: TaskID) -> bool:
        """Check whether a task with the given ID exists.

        Args:
            task_id: The task identifier to search for.

        Returns:
            True if the task exists, False otherwise.
        """
        return task_id in self.task_ids

    def get_task(self, task_id: TaskID) -> Task:
        """Retrieve the Task object by its ID.

        Args:
            task_id: The identifier of the desired task.

        Returns:
            The matching Task.

        Raises:
            ValueError: If the task_id does not exist in this workflow.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        else:
            raise ValueError(f"Task ID {task_id} does not exist")

    def add_task(
        self,
        task_option: TFCommandOptions,
        task_id: TaskID | None = None,
        *,
        pre_hooks: Sequence[PreHook] | None = None,
        post_hooks: Sequence[PostHook] | None = None,
    ) -> Workflow:
        """Add a task to this workflow and return a new Workflow instance.

        If task_id is not provided, it will be generated from the command and a sequential ID.

        Args:
            task_option: Terraform task configuration to add.
            task_id: Optional unique task identifier. If provided, must be unique within the workflow.
            pre_hooks: Optional sequence of pre-execution hooks for this task.
            post_hooks: Optional sequence of post-execution hooks for this task.

        Returns:
            A new Workflow with the added task.

        Raises:
            ValueError: If the provided task_id already exists.
        """

        task_ids = self.task_ids
        if task_id is None:
            task_id = "_".join([*task_option.command, gen_sequential_id()])

        else:
            if task_id in task_ids:
                raise ValueError(f"Task ID {task_id} already exists")

        return replace(
            self,
            tasks=tuple(
                [
                    *self.tasks,
                    Task(
                        task_id=task_id,
                        option=task_option,
                        pre_hooks=tuple(pre_hooks or ()),
                        post_hooks=tuple(post_hooks or ()),
                    ),
                ]
            ),
        )

    def change_task_option(
        self,
        task_id: TaskID,
        new_option: TFCommandOptions,
        *,
        pre_hooks: Sequence[PreHook] | None = None,
        post_hooks: Sequence[PostHook] | None = None,
    ):
        """Replace an existing task's option and hooks.

        Args:
            task_id: The ID of the task to modify.
            new_option: The new Terraform task configuration.
            pre_hooks: Optional replacement pre-hooks sequence.
            post_hooks: Optional replacement post-hooks sequence.

        Returns:
            A new Workflow with the updated task.

        Raises:
            ValueError: If the task_id does not exist.
        """
        task_index = self._get_task_index(task_id)
        new_tasks = (
            *self.tasks[:task_index],
            Task(
                task_id=task_id,
                option=new_option,
                pre_hooks=tuple(pre_hooks or ()),
                post_hooks=tuple(post_hooks or ()),
            ),
            *self.tasks[task_index + 1 :],
        )

        return replace(
            self,
            tasks=tuple(new_tasks),
        )

    def remove_task(self, task_id: TaskID) -> Workflow:
        """Remove a task from this workflow.

        Args:
            task_id: The ID of the task to remove.

        Returns:
            A new Workflow without the specified task.

        Raises:
            ValueError: If the task_id does not exist.
        """
        if not self.exist_task(task_id):
            raise ValueError(f"Task ID {task_id} does not exist")

        task_id_index = self._get_task_index(task_id)

        new_tasks = tuple(
            [*self.tasks[:task_id_index], *self.tasks[task_id_index + 1 :]]
        )

        return replace(self, tasks=new_tasks)

    def clear_tasks(self) -> Workflow:
        """Remove all tasks from this workflow.

        Returns:
            A new Workflow with an empty task list.
        """
        return replace(self, tasks=tuple())

    def _get_task_index(self, task_id: TaskID) -> int:
        """Return the index of a task by ID.

        Args:
            task_id: The ID to locate.

        Returns:
            The zero-based index of the task.

        Raises:
            ValueError: If the task_id does not exist.
        """
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
        """Run tasks asynchronously in order and return the aggregated result.

        Execution starts from the first task, or from start_task_id if provided.
        Commands are executed using the configured Terraform path within work_dir.

        Args:
            start_task_id: Optional ID of the first task to execute.
            encoding_output: If True, decode command output using the system preferred
                encoding; if a string, use it as the encoding; if False, keep bytes.

        Returns:
            WorkflowResult containing results for executed tasks (until the first failure, if any).
        """
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
    """A group of workflows with a concurrency limit.

    Attributes:
        group_id: Unique identifier of the group.
        workflows: Workflows to run as part of this group.
        max_concurrency: Maximum number of workflows to run concurrently within this group.
    """

    group_id: GroupID
    workflows: tuple[Workflow, ...]
    max_concurrency: int


@dataclass(frozen=True)
class WorkflowManager:
    """Twrapform configuration object.

    Manages one or more workflow groups and executes them with optional
    concurrency constraints. Groups are executed sequentially by default,
    while workflows inside a group may execute concurrently.
    """

    groups: tuple[WorkflowGroup, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Validate group identifiers after initialization.

        Ensures:
          - All groups have a non-None group_id
          - group_ids are unique

        Raises:
            ValueError: If a group_id is missing or duplicated.
        """
        group_ids = set(self.group_ids)

        if None in group_ids:
            raise ValueError("Group ID must be specified")

        if len(group_ids) != len(self.group_ids):
            raise ValueError("Group ID must be unique")

    @property
    def group_ids(self) -> tuple[GroupID, ...]:
        """Tuple of group IDs in their insertion order."""
        return tuple(workflow.group_id for workflow in self.groups)

    def add_workflows(
        self,
        *workflows: Workflow,
        group_id: GroupID | None = None,
        max_concurrency: int | None = None,
    ) -> WorkflowManager:
        """Add workflows and create a new workflow group.

        Args:
            *workflows: One or more Workflow instances to include in the group.
            group_id: Optional unique identifier for the group; generated if omitted.
            max_concurrency: Maximum number of workflows to run concurrently in this group.
                Defaults to the number of workflows. Must be greater than 0 if specified.

        Returns:
            A new WorkflowManager instance with the added workflow group.

        Raises:
            ValueError: If group_id already exists or max_concurrency <= 0.
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
        """Check whether a group with the given ID exists.

        Args:
            group_id: The group identifier to search for.

        Returns:
            True if the group exists, False otherwise.
        """
        group_ids = self.group_ids
        return group_id in group_ids

    def _get_group_index(self, group_id: GroupID) -> int:
        """Return the index of a group by ID.

        Args:
            group_id: The ID to locate.

        Returns:
            The zero-based index of the group.

        Raises:
            ValueError: If the group_id does not exist.
        """
        for index, group in enumerate(self.groups):
            if group.group_id == group_id:
                return index
        else:
            raise ValueError(f"Group ID {group_id} does not exist")

    def remove_group(self, group_id: GroupID) -> WorkflowManager:
        """Remove a group by ID.

        Args:
            group_id: The ID of the group to remove.

        Returns:
            A new WorkflowManager without the specified group.

        Raises:
            ValueError: If the group_id does not exist.
        """
        if not self.exist_group(group_id):
            raise ValueError(f"Group ID {group_id} does not exist")

        group_id_index = self._get_group_index(group_id)

        new_groups = tuple(
            [*self.groups[:group_id_index], *self.groups[group_id_index + 1 :]]
        )

        return replace(self, groups=new_groups)

    def clear_groups(self) -> WorkflowManager:
        """Remove all groups.

        Returns:
            A new WorkflowManager containing no groups.
        """
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
    logger: logging.Logger | None = None,
) -> tuple[CommandTaskResult | PreExecutionFailure, ...]:
    """Execute Terraform tasks sequentially, respecting resource locks.

    For each task:
      - Optionally acquire a lock to avoid concurrent access to shared resources
        (e.g., provider plugin cache for init, or the working directory).
      - Run the Terraform command as a subprocess.
      - Decode stdout/stderr if output_encoding is provided.
      - Call pre- and post-execution hooks with deep-copied arguments.
      - Stop at the first non-zero return code or at the first exception and return collected results.

    Args:
        work_dir: Directory to run Terraform commands in.
        terraform_path: Path or name of the Terraform executable.
        tasks: Ordered tasks to execute.
        env_vars: Environment variables for the subprocess; defaults to the current environment.
        output_encoding: Encoding for stdout/stderr; if None, keep as bytes.
        logger: Logger instance for diagnostic messages.

    Returns:
        A tuple of CommandTaskResult or PreExecutionFailure for each attempted task.
    """
    task_results = []

    if logger is None:
        logger = get_logger()

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

            _run_pre_hooks(work_dir, task.task_id, task.option, task.pre_hooks)

            if isinstance(task.option, InitTaskOptions):
                resource_id = str(
                    env_vars.get("TF_PLUGIN_CACHE_DIR")
                    or _provider_cache_dir
                    or work_dir
                )
            else:
                resource_id = str(work_dir)

            async with _lock_manager_instance.get_lock(resource_id):
                proc = await asyncio.create_subprocess_exec(
                    terraform_path,
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env_vars,
                )

            del resource_id
            stdout, stderr = await proc.communicate()
            return_code = await proc.wait()

            if output_encoding is not None:
                try:
                    stdout = stdout.decode(output_encoding)
                    stderr = stderr.decode(output_encoding)
                except UnicodeDecodeError as e:
                    logger.warning("[%s] Failed encoding output: %s", task.task_id, e)

            task_result = CommandTaskResult(
                task_id=task.task_id,
                task_option=task.option,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
            )

            _run_post_hooks(
                work_dir, task.task_id, task.option, task.post_hooks, task_result
            )

            task_results.append(task_result)

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
    """Normalize the encoding_output parameter to an explicit encoding or None.

    Args:
        encoding_output: If True, use locale.getpreferredencoding(); if str, use as-is;
            if False or None, return None.

    Returns:
        The encoding name to use, or None to keep bytes.
    """
    if isinstance(encoding_output, bool):
        if encoding_output:
            return locale.getpreferredencoding()

        return None

    return encoding_output


def _run_pre_hooks(
    work_dir: os.PathLike[str] | str,
    task_id: TaskID,
    option: TFCommandOptions,
    hooks: Sequence[PreHook],
):
    """Execute pre-execution hooks safely.

    Each hook is called with (work_dir, deepcopy(option)).
    Exceptions raised by hooks are caught and logged; execution continues.

    Args:
        work_dir: The working directory path.
        task_id: The associated task identifier (for logging).
        option: The task option to pass to hooks (deep-copied).
        hooks: A sequence of pre-execution hooks.
    """
    logger = get_logger()

    for i, hook in enumerate(hooks):
        try:
            hook(work_dir, copy.deepcopy(option))
        except Exception as e:
            logger.warning("[%s] Failed running pre hook %s: %s", task_id, i, repr(e))


def _run_post_hooks(
    work_dir: os.PathLike[str] | str,
    task_id: TaskID,
    option: TFCommandOptions,
    hooks: Sequence[PostHook],
    task_result: TaskResult,
):
    """Execute post-execution hooks safely.

    Each hook is called with (work_dir, deepcopy(option), deepcopy(task_result)).
    Exceptions raised by hooks are caught and logged; execution continues.

    Args:
        work_dir: The working directory path.
        task_id: The associated task identifier (for logging).
        option: The task option to pass to hooks (deep-copied).
        hooks: A sequence of post-execution hooks.
        task_result: The result object to pass to hooks (deep-copied).
    """
    logger = get_logger()

    for i, hook in enumerate(hooks):
        try:
            hook(work_dir, copy.deepcopy(option), copy.deepcopy(task_result))
        except Exception as e:
            logger.warning("[%s] Failed running post hook %s: %s", task_id, i, repr(e))
