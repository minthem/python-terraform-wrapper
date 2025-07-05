from __future__ import annotations

import asyncio
import locale
import logging
import os
from dataclasses import dataclass, field, replace

from .logging import get_logger
from .options import SupportedTerraformTask
from .result import PreExecutionFailure, TwrapformCommandTaskResult, TwrapformResult
from .task import Task, TaskID, gen_sequential_id


@dataclass(frozen=True)
class Workflow:
    """Twrapform configuration object."""

    work_dir: os.PathLike[str] | str
    terraform_path: os.PathLike[str] | str = "terraform"
    tasks: tuple[Task, ...] = field(default_factory=tuple)

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
        """Add a task to the Twrapform object."""

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
            task for i, task in enumerate(self.tasks) if i != task_id_index
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
    ) -> TwrapformResult:
        """Run all tasks asynchronously."""

        env_vars = os.environ.copy()
        encoding: str | None = None

        if start_task_id is not None:
            start_index = self._get_task_index(start_task_id)
        else:
            start_index = 0

        if isinstance(encoding_output, bool):
            if encoding_output:
                encoding = locale.getpreferredencoding()
        else:
            encoding = encoding_output

        return await _execute_terraform_tasks(
            work_dir=self.work_dir,
            terraform_path=self.terraform_path,
            tasks=self.tasks[start_index:],
            env_vars=env_vars,
            output_encoding=encoding,
        )


async def _execute_terraform_tasks(
    work_dir: os.PathLike[str] | str,
    terraform_path: os.PathLike[str] | str,
    tasks: tuple[Task, ...],
    env_vars: dict[str, str] | None = None,
    output_encoding: str | None = None,
    logger: logging.Logger = get_logger(),
) -> TwrapformResult:
    task_results = []

    if env_vars is None:
        env_vars = os.environ.copy()

    for task in tasks:
        try:
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
                TwrapformCommandTaskResult(
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

    return TwrapformResult(task_results=tuple(task_results))
