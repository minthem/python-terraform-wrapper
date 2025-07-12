from .common import GroupID, TaskID


class TwrapformError(Exception):
    def __init__(self, message: str):
        self.message = message

        super().__init__(self.message)


class TwrapformPreconditionError(TwrapformError):
    def __init__(
        self,
        task_id: TaskID,
        exc: Exception,
    ):
        self.task_id = task_id
        self.original_exception = exc

        super().__init__(
            message=f"Twrapform precondition error, original error = {repr(self.original_exception)}",
        )


class TwrapformTaskError(TwrapformError):
    def __init__(
        self,
        task_id: TaskID,
        return_code: int,
        stdout: str | bytes,
        stderr: str | bytes,
    ):
        self.task_id = task_id
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

        super().__init__(
            message=f"Terraform command failed with return code {return_code}, stdout: {stdout}, stderr: {stderr}",
        )


class TwrapformGroupExecutionError(TwrapformError):
    def __init__(self, group_id: GroupID, errors: tuple[TwrapformError, ...]):
        self.errors = errors
        self.group_id = group_id

        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"Group {self.group_id} failed with {len(self.errors)} error(s):\n"
            + "\n".join(f"  - {type(e).__name__}: {e}" for e in self.errors)
        )


class WorkflowManagerExecutionError(TwrapformError):
    def __init__(self, group_errors: tuple[TwrapformGroupExecutionError, ...]):
        self.group_errors = group_errors
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"{len(self.group_errors)} group(s) failed during execution:\n"
            + "\n\n".join(str(group_error) for group_error in self.group_errors)
        )
