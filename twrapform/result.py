from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .common import GroupID, TaskID, WorkflowID
from .exception import (
    TwrapformError,
    TwrapformGroupExecutionError,
    TwrapformPreconditionError,
    TwrapformTaskError,
    WorkflowManagerExecutionError,
)
from .options import SupportedTerraformTask


@dataclass(frozen=True)
class TaskResult(ABC):
    task_id: TaskID
    task_option: SupportedTerraformTask

    @abstractmethod
    def is_success(self) -> bool: ...

    @abstractmethod
    def summary(self) -> str: ...

    @abstractmethod
    def raise_on_error(self): ...


@dataclass(frozen=True)
class CommandTaskResult(TaskResult):
    return_code: int
    stdout: str | bytes
    stderr: str | bytes

    def is_success(self) -> bool:
        return self.return_code == 0

    def summary(self) -> str:
        return f"[{self.task_id}] Completed with code {self.return_code}"

    def raise_on_error(self):
        if not self.is_success():
            raise TwrapformTaskError(
                task_id=self.task_id,
                return_code=self.return_code,
                stdout=self.stdout,
                stderr=self.stderr,
            )


@dataclass(frozen=True)
class PreExecutionFailure(TaskResult):
    original_error: Exception

    def is_success(self) -> bool:
        return False

    def summary(self) -> str:
        return (
            f"[{self.task_id}] Failed before execution: ({repr(self.original_error)})"
        )

    def raise_on_error(self):
        raise TwrapformPreconditionError(
            task_id=self.task_id,
            exc=self.original_error,
        )


@dataclass(frozen=True)
class WorkflowResult:
    """Twrapform task result object."""

    workflow_id: WorkflowID
    task_results: tuple[CommandTaskResult | TwrapformPreconditionError, ...] = field(
        default_factory=tuple
    )

    def raise_on_error(self):
        """Raise an exception if any task failed."""
        for task_result in self.task_results:
            task_result.raise_on_error()

    def get_result(
        self, task_id: TaskID
    ) -> CommandTaskResult | TwrapformPreconditionError:
        """Get a task result by its ID."""
        for task_result in self.task_results:
            if task_result.task_id == task_id:
                return task_result
        else:
            raise ValueError(f"No task result for task_id {task_id}")

    @property
    def result_count(self) -> int:
        """Return the number of task results."""
        return len(self.task_results)

    @property
    def success_count(self) -> int:
        """Return the number of success results."""
        return len([result for result in self.task_results if result.is_success()])

    def get_success_tasks(self) -> tuple[CommandTaskResult, ...]:
        """Return all task results."""
        return tuple(
            task_result for task_result in self.task_results if task_result.is_success()
        )

    def is_all_success(self) -> bool:
        """Return True if all task results are success."""
        return self.result_count == self.success_count


@dataclass(frozen=True)
class WorkflowGroupResult:
    group_id: GroupID
    workflow_results: tuple[WorkflowResult, ...]

    def raise_on_error(self):
        errors: list[TwrapformError] = []

        for wf_result in self.workflow_results:
            for task_result in wf_result.task_results:
                try:
                    task_result.raise_on_error()
                except TwrapformError as e:
                    errors.append(e)

        if 0 < len(errors):
            raise TwrapformGroupExecutionError(
                group_id=self.group_id, errors=tuple(errors)
            )

    @property
    def success_workflow_count(self) -> int:
        return sum(1 for wf in self.workflow_results if wf.is_all_success())

    def is_all_success(self) -> bool:
        return self.success_workflow_count == len(self.workflow_results)

    def get_workflow_result(self, workflow_id: WorkflowID) -> WorkflowResult:
        for wf in self.workflow_results:
            if wf.workflow_id == workflow_id:
                return wf
        else:
            raise ValueError(f"No workflow result for workflow_id {workflow_id}")


@dataclass(frozen=True)
class WorkflowManagerResult:
    group_results: tuple[WorkflowGroupResult, ...]

    def raise_on_error(self):
        errors: list[TwrapformGroupExecutionError] = []

        for group_result in self.group_results:
            try:
                group_result.raise_on_error()
            except TwrapformGroupExecutionError as e:
                errors.append(e)

        if 0 < len(errors):
            raise WorkflowManagerExecutionError(tuple(errors))

    @property
    def success_count(self) -> int:
        return sum(1 for gr in self.group_results if gr.is_all_success())

    def is_all_success(self) -> bool:
        return self.success_count == len(self.group_results)

    def get_group_result(self, group_id: GroupID) -> WorkflowGroupResult:
        for gr in self.group_results:
            if gr.group_id == group_id:
                return gr
        else:
            raise ValueError(f"No group result for group_id {group_id}")
