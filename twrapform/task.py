from __future__ import annotations

import time
from typing import NamedTuple

from .options import SupportedTerraformTask

TaskID = int | str


class Task(NamedTuple):
    task_id: TaskID
    option: SupportedTerraformTask


def gen_sequential_id() -> TaskID:
    result = str(int(time.time()))
    return result
