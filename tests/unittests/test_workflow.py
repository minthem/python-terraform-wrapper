import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from twrapform import Task, Workflow
from twrapform.exception import TwrapformPreconditionError, TwrapformTaskError
from twrapform.options import (
    ApplyTaskOptions,
    InitTaskOptions,
    OutputTaskOptions,
    PlanTaskOptions,
)
from twrapform.result import CommandTaskResult, PreExecutionFailure


class TestTwrapformConstructor:
    """Test cases for TwrapformTask class."""

    def test_create_instance(self):
        workdir = os.getcwd()
        t = Workflow(work_dir=workdir)

        assert workdir == t.work_dir
        assert "terraform" == t.terraform_path
        assert len(t.tasks) == 0

    def test_create_instance_specific_terraform_path(self):
        workdir = os.path.join(os.getcwd(), "work")
        path = "/home/user/.local/bin/terraform"
        t = Workflow(work_dir=workdir, terraform_path=path)

        assert workdir == t.work_dir
        assert path == t.terraform_path
        assert len(t.tasks) == 0

    def test_create_instance_specific_terraform_path(self):
        workdir = os.path.join(os.getcwd(), "work")
        path = "/home/user/.local/bin/terraform"
        t = Workflow(work_dir=workdir, terraform_path=path)

        assert workdir == t.work_dir
        assert path == t.terraform_path
        assert len(t.tasks) == 0

    def test_create_instance_specific_tasks(self):
        workdir = os.path.join(os.getcwd(), "work")
        tasks = tuple(
            [
                Task(task_id=1, option=InitTaskOptions()),
                Task(task_id=2, option=PlanTaskOptions()),
                Task(task_id=3, option=ApplyTaskOptions()),
            ]
        )
        t = Workflow(
            work_dir=workdir,
            tasks=tasks,
        )

        assert t.work_dir == workdir
        assert t.terraform_path == "terraform"
        assert len(t.tasks) == 3

    def test_create_instance_task_id_is_none(self):
        workdir = os.path.join(os.getcwd(), "work")
        tasks = tuple(
            [
                Task(task_id=None, option=InitTaskOptions()),
                Task(task_id=2, option=PlanTaskOptions()),
            ]
        )
        with pytest.raises(ValueError, match="Task ID must be specified"):
            Workflow(
                work_dir=workdir,
                tasks=tasks,
            )

    def test_create_instance_dup_task_id(self):
        workdir = os.path.join(os.getcwd(), "work")
        tasks = tuple(
            [
                Task(task_id=2, option=InitTaskOptions()),
                Task(task_id=2, option=PlanTaskOptions()),
            ]
        )
        with pytest.raises(ValueError, match="Task ID must be unique"):
            Workflow(
                work_dir=workdir,
                tasks=tasks,
            )


class TestTwrapformAddTask:
    def test_add_task_creates_task_with_default_id(self):
        """タスクIDが指定されない場合、デフォルトのIDでタスクを作成するテスト"""
        twrapform = Workflow(work_dir=Path("/tmp"))
        new_option = InitTaskOptions()
        updated_twrapform = twrapform.add_task(task_option=new_option)

        assert len(updated_twrapform.tasks) == 1
        assert updated_twrapform.tasks[0].task_id.startswith("init_")
        assert updated_twrapform.tasks[0].option == new_option

    def test_add_task_creates_task_with_specified_id(self):
        """指定されたタスクIDでタスクを作成するテスト"""
        twrapform = Workflow(work_dir=Path("/tmp"))
        new_option = InitTaskOptions()
        updated_twrapform = twrapform.add_task(task_option=new_option, task_id=42)

        assert len(updated_twrapform.tasks) == 1
        assert updated_twrapform.tasks[0].task_id == 42
        assert updated_twrapform.tasks[0].option == new_option

    def test_add_task_raises_error_for_duplicate_id(self):
        """重複するタスクIDが指定された際にエラーを投げることを確認"""
        twrapform = Workflow(work_dir=Path("/tmp"))
        new_option = InitTaskOptions()
        twrapform = twrapform.add_task(task_option=new_option, task_id=1)

        with pytest.raises(ValueError, match="Task ID 1 already exists"):
            twrapform.add_task(task_option=PlanTaskOptions(), task_id=1)


class TestTwrapformRemoveTask:
    def test_remove_task_removes_specified_task(self):
        """指定されたタスクが正しく削除されることを確認"""
        task1 = Task(task_id=1, option=InitTaskOptions())
        task2 = Task(task_id=2, option=PlanTaskOptions())
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=2, task_option=PlanTaskOptions())
        )

        updated_twrapform = twrapform.remove_task(task_id=1)

        assert len(updated_twrapform.tasks) == 1
        assert updated_twrapform.tasks[0].task_id == 2

    def test_remove_task_raises_error_for_nonexistent_task(self):
        """存在しないタスクIDを指定するとエラーを投げることを確認"""
        twrapform = Workflow(work_dir=Path("/tmp")).add_task(
            task_id=1, task_option=InitTaskOptions()
        )

        with pytest.raises(ValueError, match="Task ID 99 does not exist"):
            twrapform.remove_task(task_id=99)

    def test_remove_task_does_not_modify_original_instance(self):
        """remove_taskが元のインスタンスを変更しないことを確認"""
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=2, task_option=PlanTaskOptions())
        )

        _ = twrapform.remove_task(task_id=1)

        assert len(twrapform.tasks) == 2
        assert twrapform.tasks[0].task_id == 1
        assert twrapform.tasks[1].task_id == 2

    def test_remove_task_handles_empty_task_list(self):
        """タスクが存在しない状態でremove_taskがエラーを投げることを確認"""
        twrapform = Workflow(work_dir=Path("/tmp"), tasks=())

        with pytest.raises(ValueError, match="Task ID 1 does not exist"):
            twrapform.remove_task(task_id=1)


class TestTwrapformClearTasks:
    def test_clear_tasks_removes_all_tasks(self):
        """全てのタスクが正しく削除されることを確認"""
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=2, task_option=PlanTaskOptions())
        )

        updated_twrapform = twrapform.clear_tasks()

        assert len(updated_twrapform.tasks) == 0

    def test_clear_tasks_does_not_modify_original_instance(self):
        """元のインスタンスが変更されていないことを確認"""
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=2, task_option=PlanTaskOptions())
        )

        _ = twrapform.clear_tasks()

        # 元のインスタンスはそのまま
        assert len(twrapform.tasks) == 2
        assert twrapform.tasks[0].task_id == 1
        assert twrapform.tasks[1].task_id == 2


class TestTwrapformGetTask:
    def test_get_task_returns_correct_task(self):
        """指定されたタスクIDが正しいタスクを返すことを確認"""
        option = PlanTaskOptions()
        task_id = 2
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=task_id, task_option=option)
        )

        result = twrapform.get_task(task_id=task_id)
        assert result.task_id == task_id
        assert result.option == option

    def test_get_task_raises_error_for_nonexistent_task(self):
        """存在しないタスクIDを指定するとエラーを投げることを確認"""
        twrapform = Workflow(work_dir=Path("/tmp")).add_task(
            task_id=1, task_option=ApplyTaskOptions()
        )

        with pytest.raises(ValueError, match="Task ID 99 does not exist"):
            twrapform.get_task(task_id=99)

    def test_get_task_handles_string_task_id(self):
        """文字列のタスクIDを指定した場合でも正しく動作することを確認"""
        option = ApplyTaskOptions()
        task_id = "test_string"
        twrapform = (
            Workflow(work_dir=Path("/tmp"))
            .add_task(task_id=1, task_option=InitTaskOptions())
            .add_task(task_id=task_id, task_option=option)
        )

        result = twrapform.get_task(task_id=task_id)
        assert result.task_id == task_id
        assert result.option == option

    def test_get_task_raises_error_for_empty_task_list(self):
        """タスクが存在しない場合にエラーを投げることを確認"""
        twrapform = Workflow(work_dir=Path("/tmp"), tasks=())

        with pytest.raises(ValueError, match="Task ID 1 does not exist"):
            twrapform.get_task(task_id=1)


class TestTwrapformExecute:

    # pytest.mark.asyncio を使った非同期テスト
    @pytest.mark.asyncio
    async def test_run_await_executes_all_tasks_successfully(self):
        """すべてのタスクが成功する場合の動作を確認"""
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions())
            .add_task(task_option=PlanTaskOptions())
            .add_task(task_option=ApplyTaskOptions())
            .add_task(task_option=OutputTaskOptions())
        )

        # プロセス作成をモック
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"success", b"")
            mock_proc.wait.return_value = 0
            mock_exec.return_value = mock_proc

            results = await twrapform.execute()

            assert results.result_count == len(twrapform.tasks)
            for task, result in zip(twrapform.tasks, results.task_results):
                assert task.task_id == result.task_id
                assert task.option == result.task_option
                assert isinstance(result, CommandTaskResult) is True
                assert result.is_success() is True
                assert result.summary() == f"[{task.task_id}] Completed with code 0"
                try:
                    result.raise_on_error()
                except Exception as e:
                    pytest.fail(f"An unexpected exception was raised: {e} (result)")

            try:
                results.raise_on_error()
            except Exception as e:
                pytest.fail(f"An unexpected exception was raised: {e} (results)")

    @pytest.mark.asyncio
    async def test_run_await_encoding_error(self):
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions())
            .add_task(task_option=PlanTaskOptions())
            .add_task(task_option=ApplyTaskOptions())
            .add_task(task_option=OutputTaskOptions())
        )

        # プロセス作成をモック
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                "あああああああああああああ".encode(),
                b"",
            )
            mock_proc.wait.return_value = 0
            mock_exec.return_value = mock_proc

            results = await twrapform.execute(encoding_output="ascii")

            assert results.result_count == len(twrapform.tasks)
            for task, result in zip(twrapform.tasks, results.task_results):
                assert task.task_id == result.task_id
                assert task.option == result.task_option
                assert isinstance(result, CommandTaskResult) is True
                assert result.is_success() is True
                assert result.summary() == f"[{task.task_id}] Completed with code 0"
                try:
                    result.raise_on_error()
                except Exception as e:
                    pytest.fail(f"An unexpected exception was raised: {e} (result)")

                assert isinstance(result.stdout, bytes) is True
                assert isinstance(result.stderr, bytes) is True

            try:
                results.raise_on_error()
            except Exception as e:
                pytest.fail(f"An unexpected exception was raised: {e} (results)")

    @pytest.mark.asyncio
    async def test_run_await_stops_on_failure(self):
        """途中でエラーが発生した場合、処理が停止することを確認"""
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions())
            .add_task(task_option=PlanTaskOptions())
            .add_task(task_option=ApplyTaskOptions())
            .add_task(task_option=OutputTaskOptions())
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc1 = AsyncMock()
            mock_proc1.communicate.return_value = (b"success", b"")
            mock_proc1.wait.return_value = 0

            mock_proc2 = AsyncMock()
            mock_proc2.communicate.return_value = (b"", b"error")
            mock_proc2.wait.return_value = 1

            mock_exec.side_effect = [mock_proc1, mock_proc1, mock_proc2]

            results = await twrapform.execute(encoding_output=True)

            assert results.result_count == 3
            for task, result in zip(twrapform.tasks[:2], results.task_results):
                assert task.task_id == result.task_id
                assert task.option == result.task_option
                assert isinstance(result, CommandTaskResult) is True
                assert result.is_success() is True
                assert result.summary() == f"[{task.task_id}] Completed with code 0"
                try:
                    result.raise_on_error()
                except Exception as e:
                    pytest.fail(f"An unexpected exception was raised: {e}")

            last_task = twrapform.tasks[2]
            fail_task = results.task_results[-1]

            assert last_task.task_id == fail_task.task_id
            assert last_task.option == fail_task.task_option
            assert isinstance(fail_task, CommandTaskResult) is True
            assert fail_task.is_success() is False
            assert fail_task.summary() == f"[{fail_task.task_id}] Completed with code 1"

            with pytest.raises(
                TwrapformTaskError,
                match=f"Terraform command failed with return code 1, stdout: , stderr: error",
            ):
                fail_task.raise_on_error()

            with pytest.raises(
                TwrapformTaskError,
                match=f"Terraform command failed with return code 1, stdout: , stderr: error",
            ):
                results.raise_on_error()

    @pytest.mark.asyncio
    async def test_run_await_handles_exceptions(self):
        """タスク実行中に例外が発生した場合をテスト"""
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions())
            .add_task(task_option=PlanTaskOptions())
            .add_task(task_option=ApplyTaskOptions())
            .add_task(task_option=OutputTaskOptions())
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc1 = AsyncMock()
            mock_proc1.communicate.return_value = (b"success", b"")
            mock_proc1.wait.return_value = 0

            mock_proc2 = AsyncMock()
            mock_proc2.side_effect = Exception("Mock error")

            mock_exec.side_effect = [mock_proc1, mock_proc2]
            results = await twrapform.execute()

            assert results.result_count == 2
            for task, result in zip(twrapform.tasks[:1], results.task_results):
                assert task.task_id == result.task_id
                assert task.option == result.task_option
                assert isinstance(result, CommandTaskResult) is True
                assert result.is_success() is True
                assert result.summary() == f"[{task.task_id}] Completed with code 0"
                try:
                    result.raise_on_error()
                except Exception as e:
                    pytest.fail(f"An unexpected exception was raised: {e}")

            last_task = twrapform.tasks[1]
            fail_task = results.task_results[-1]

            assert last_task.task_id == fail_task.task_id
            assert last_task.option == fail_task.task_option
            assert isinstance(fail_task, PreExecutionFailure) is True
            assert fail_task.is_success() is False
            assert fail_task.summary().startswith(
                f"[{fail_task.task_id}] Failed before execution:"
            )

            with pytest.raises(
                TwrapformPreconditionError,
                match=f"Twrapform precondition error, original error = .+",
            ):
                fail_task.raise_on_error()

            with pytest.raises(
                TwrapformPreconditionError,
                match=f"Twrapform precondition error, original error = .+",
            ):
                results.raise_on_error()

    @pytest.mark.asyncio
    async def test_run_await_executes_specific_task_id_successfully(self):
        """タスクIDを指定した実行"""
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions(), task_id="test1")
            .add_task(task_option=PlanTaskOptions(), task_id="test2")
            .add_task(task_option=ApplyTaskOptions(), task_id="test3")
            .add_task(task_option=OutputTaskOptions(), task_id="test4")
        )

        # プロセス作成をモック
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"success", b"")
            mock_proc.wait.return_value = 0
            mock_exec.return_value = mock_proc

            results = await twrapform.execute(start_task_id="test3")

            assert results.result_count == 2
            for task, result in zip(twrapform.tasks[2:], results.task_results):
                assert task.task_id == result.task_id
                assert task.option == result.task_option
                assert isinstance(result, CommandTaskResult) is True
                assert result.is_success() is True
                assert result.summary() == f"[{task.task_id}] Completed with code 0"
                try:
                    result.raise_on_error()
                except Exception as e:
                    pytest.fail(f"An unexpected exception was raised: {e} (result)")

            try:
                results.raise_on_error()
            except Exception as e:
                pytest.fail(f"An unexpected exception was raised: {e} (results)")

    @pytest.mark.asyncio
    async def test_run_await_raises_error_for_nonexistent_task_id(self):
        """存在しないタスクIDを指定した場合のエラーを確認"""
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_option=InitTaskOptions(), task_id="test1")
            .add_task(task_option=PlanTaskOptions(), task_id="test2")
        )

        with pytest.raises(ValueError, match="Task ID 99 does not exist"):
            await twrapform.execute(start_task_id=99)

    @pytest.mark.asyncio
    async def test_run_await_handles_empty_task_list_with_task_id(self):
        """タスクが登録されていない状態でタスクIDを指定して実行した場合"""
        twrapform = Workflow(work_dir="/tmp")

        with pytest.raises(ValueError, match="Task ID 1 does not exist"):
            await twrapform.execute(start_task_id=1)

    @pytest.mark.asyncio
    async def test_run_await_handles_empty_task_list_without_task_id(self):
        """タスクが登録されていない状態でタスクIDなしで実行した場合"""
        twrapform = Workflow(work_dir="/tmp")

        # 実行結果が空であることを確認
        result = await twrapform.execute()
        assert len(result.task_results) == 0


class TestTwrapformChangeTaskOption:
    @pytest.fixture
    def twrapform(self):
        # 初期状態のタスクを設定
        twrapform = (
            Workflow(work_dir="/tmp")
            .add_task(task_id=1, task_option=InitTaskOptions(backend=True))
            .add_task(task_id=2, task_option=PlanTaskOptions())
            .add_task(task_id=3, task_option=ApplyTaskOptions())
            .add_task(task_id=4, task_option=OutputTaskOptions())
        )
        return twrapform

    def test_change_task_option_success(self, twrapform):
        # 新しいオプションを準備
        new_option = PlanTaskOptions(parallelism=5)

        # タスクオプションを変更
        updated_obj = twrapform.change_task_option(task_id=2, new_option=new_option)

        # 検証: タスクが正しく更新されているか
        assert len(updated_obj.tasks) == len(twrapform.tasks)
        assert updated_obj.get_task(2).option == new_option

    def test_change_task_option_invalid_task_id(self, twrapform):
        # 存在しない `task_id` を使用
        with pytest.raises(ValueError, match="Task ID 999 does not exist"):
            twrapform.change_task_option(task_id=999, new_option=ApplyTaskOptions())

    def test_change_task_option_blank_task(self):
        twrapform = Workflow(work_dir="/tmp")
        with pytest.raises(ValueError, match="Task ID 999 does not exist"):
            twrapform.change_task_option(task_id=999, new_option=ApplyTaskOptions())
