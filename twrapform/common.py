from __future__ import annotations

import time
from typing import NamedTuple

from .options import SupportedTerraformTask

TaskID = int | str
WorkflowID = int | str
GroupID = int | str


class Task(NamedTuple):
    task_id: TaskID
    option: SupportedTerraformTask


def gen_sequential_id() -> TaskID:
    result = hex(int(time.time() * (10**8)))[2:]
    return result


def gen_workflow_id() -> WorkflowID:
    return "workflow_" + gen_sequential_id()


def gen_group_id() -> GroupID:
    return "group_" + gen_sequential_id()
