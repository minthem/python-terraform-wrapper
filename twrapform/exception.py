from .task import TaskID


class TwrapformError(Exception):
    def __init__(self, task_id: TaskID, message: str):
        self.task_id = task_id
        self.message = message

        super().__init__(self.message)


class TwrapformPreconditionError(TwrapformError):
    def __init__(
        self,
        task_id: TaskID,
        exc: Exception,
    ):
        self.original_exception = exc

        super().__init__(
            task_id=task_id,
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
            task_id=task_id,
            message=f"Terraform command failed with return code {return_code}, stdout: {stdout}, stderr: {stderr}",
        )
