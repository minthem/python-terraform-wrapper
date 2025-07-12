# twrapform

A Python library for running Terraform commands from Python with asynchronous workflow management.

## ✨ Features

- Run Terraform commands natively from Python
- Asynchronous execution using `asyncio`
- Immutable task definitions with unique IDs
- Grouped workflow orchestration
- Granular error handling with structured results

## 📦 Requirements

- Python 3.10+
- Terraform installed and available in your system's `PATH`

## 🔧 Installation

```bash
pip install twrapform
```

## 🚀 Usage Examples

### Run a single workflow with chained Terraform tasks
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

### Manage multiple workflows in a group using `WorkflowManager`
```python
from twrapform import Workflow, WorkflowManager
from twrapform.options import InitTaskOptions
import asyncio

async def main():
    # Define workflows with Terraform initialization tasks
    workflow1 = Workflow(work_dir="infra/project1").add_task(InitTaskOptions())
    workflow2 = Workflow(work_dir="infra/project2").add_task(InitTaskOptions())
    
    # Add workflows into a group and initialize the manager
    manager = WorkflowManager().add_workflows(workflow1, workflow2, group_id="init-group")
    
    # Run the grouped workflows asynchronously
    result = await manager.execute()
    
    # Display summaries for successfully completed tasks
    for group_result in result.group_results:
        for wf_result in group_result.workflow_results:
            for task in wf_result.get_success_tasks():
                print(task.summary())

                
if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ Supported Terraform Commands
twrapform currently supports the following Terraform commands:
* `terraform init`
* `terraform plan`
* `terraform apply`
* `terraform output`
* `terraform workspace select`