# twrapform

A Python library for running Terraform commands from Python with async support.

## Features

- Execute Terraform commands from Python code
- Asynchronous execution support
- Task management with unique IDs
- Customizable task options
- Error handling for Terraform operations

## Requirements

- Python 3.10 or higher
- Terraform installed on your system

## Installation

Using pip:
```shell
pip install twrapform
```

## Usage Example

```python
import asyncio

from twrapform import Workflow
from twrapform.exception import TwrapformError
from twrapform.options import InitTaskOptions, PlanTaskOptions, ApplyTaskOptions, OutputTaskOptions


async def main():
    # Create an instance of Twrapform
    twrap = Workflow(work_dir="/terraform_rootpath")

    # Add Terraform tasks one by one
    twrap = twrap.add_task(InitTaskOptions())

    # Chain multiple tasks
    twrap = (
        twrap
        .add_task(PlanTaskOptions(var={"var1": 1}))
        .add_task(ApplyTaskOptions(var={"var1": 1}))
        .add_task(OutputTaskOptions(json=True))
    )

    # Execute all tasks
    results = await twrap.execute()

    try:
        # Raise errors if any task fails
        results.raise_on_error()
    except TwrapformError as e:
        print(f"Error occurred: {e.message}")

    # Output results for successful tasks
    for success_task in results.get_success_tasks():
        print(success_task.stdout)


if __name__ == "__main__":
    asyncio.run(main())

```

## Supported Commands
twrapform currently supports the following Terraform commands:
* `terraform init`
* `terraform plan`
* `terraform apply`
* `terraform output`
* `terraform workspace select`