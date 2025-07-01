from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from .result import (
    PreExecutionFailure,
    TwrapformCommandTaskResult,
    TwrapformResult,
)
from .task import SupportedTerraformTask


class TwrapformTask(NamedTuple):
    task_id: str | int
    option: SupportedTerraformTask


@dataclass(frozen=True)
class Twrapform:
    """Twrapform configuration object."""

    work_dir: os.PathLike[str] | str
    terraform_path: os.PathLike[str] | str = "terraform"
    tasks: tuple[TwrapformTask, ...] = field(default_factory=tuple)

    def __post_init__(self):
        task_ids = set(self.task_ids)

        if None in task_ids:
            raise ValueError("Task ID must be specified")

        if len(task_ids) != len(self.tasks):
            raise ValueError("Task ID must be unique")

    @property
    def task_ids(self) -> tuple[int | str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def exist_task(self, task_id: int | str) -> bool:
        return task_id in self.task_ids

    def get_task(self, task_id: int | str) -> TwrapformTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        else:
            raise ValueError(f"Task ID {task_id} does not exist")

    def add_task(
        self, task_option: SupportedTerraformTask, task_id: str | int | None = None
    ) -> Twrapform:
        """Add a task to the Twrapform object."""

        task_ids = self.task_ids
        if task_id is None:
            task_id = len(self.tasks)
            while task_id in task_ids:
                task_id += 1
        else:
            if task_id in task_ids:
                raise ValueError(f"Task ID {task_id} already exists")

        return replace(
            self,
            tasks=tuple(
                [*self.tasks, TwrapformTask(task_id=task_id, option=task_option)]
            ),
        )

    def change_task_option(
        self, task_id: int | str, new_option: SupportedTerraformTask
    ):
        """Change the option of a task."""
        task_index = self._get_task_index(task_id)
        new_tasks = (
            *self.tasks[:task_index],
            TwrapformTask(task_id=task_id, option=new_option),
            *self.tasks[task_index + 1 :],
        )

        return replace(
            self,
            tasks=tuple(new_tasks),
        )

    def remove_task(self, task_id: str | int) -> Twrapform:
        """Remove a task from the Twrapform object."""

        task_ids = self.task_ids
        if not self.exist_task(task_id):
            raise ValueError(f"Task ID {task_id} does not exist")

        task_id_index = self._get_task_index(task_id)

        new_tasks = tuple(
            task for i, task in enumerate(self.tasks) if i != task_id_index
        )

        return replace(self, tasks=new_tasks)

    def clear_tasks(self) -> Twrapform:
        """Remove all tasks from the Twrapform object."""
        return replace(self, tasks=tuple())

    def _get_task_index(self, task_id: str | int) -> int:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return index
        else:
            raise ValueError(f"Task ID {task_id} does not exist")

    async def run_await(
        self, start_task_id: str | int | None = None
    ) -> TwrapformResult:
        """Run all tasks asynchronously."""

        task_results = []
        env_vars = os.environ.copy()

        if start_task_id is not None:
            start_index = self._get_task_index(start_task_id)
        else:
            start_index = 0

        for task in self.tasks[start_index:]:
            try:
                cmd_args = (
                    f"-chdir={self.work_dir}",
                    *task.option.convert_command_args(),
                )
                proc = await asyncio.create_subprocess_exec(
                    self.terraform_path,
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env_vars,
                )

                stdout, stderr = await proc.communicate()
                return_code = await proc.wait()
                stdout = stdout.decode()
                stderr = stderr.decode()

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
