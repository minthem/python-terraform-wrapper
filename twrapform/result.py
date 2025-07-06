from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .exception import TwrapformPreconditionError, TwrapformTaskError
from .options import SupportedTerraformTask
from .task import TaskID


@dataclass(frozen=True)
class TwrapformTaskResult(ABC):
    task_id: TaskID
    task_option: SupportedTerraformTask

    @abstractmethod
    def is_success(self) -> bool: ...

    @abstractmethod
    def summary(self) -> str: ...

    @abstractmethod
    def raise_on_error(self): ...


@dataclass(frozen=True)
class TwrapformCommandTaskResult(TwrapformTaskResult):
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
class PreExecutionFailure(TwrapformTaskResult):
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
class TwrapformResult:
    """Twrapform task result object."""

    task_results: tuple[
        TwrapformCommandTaskResult | TwrapformPreconditionError, ...
    ] = field(default_factory=tuple)

    def raise_on_error(self):
        """Raise an exception if any task failed."""
        for task_result in self.task_results:
            task_result.raise_on_error()

    def get_result(
        self, task_id: TaskID
    ) -> TwrapformCommandTaskResult | TwrapformPreconditionError:
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

    def get_success_tasks(self) -> tuple[TwrapformCommandTaskResult, ...]:
        """Return all task results."""
        return tuple(
            task_result for task_result in self.task_results if task_result.is_success()
        )
