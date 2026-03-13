"""
Submit tests to Databricks via Jobs API.
"""
import os
import time
import base64
import pathlib
import sys

# Load credentials
env_path = pathlib.Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

repo_root = pathlib.Path(__file__).parent
workspace_dir = "/Workspace/insurance-lda-risk-tests"

def upload_file(local_path: pathlib.Path, ws_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    try:
        w.workspace.mkdirs(str(pathlib.Path(ws_path).parent))
    except Exception:
        pass
    w.workspace.import_(
        path=ws_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )

# Upload only source/test/pyproject files (skip .venv)
for src_file in sorted(repo_root.rglob("*.py")):
    rel = src_file.relative_to(repo_root)
    parts = rel.parts
    # Skip virtual env, .git, and this script itself
    if any(p in (".venv", ".git", "__pycache__") for p in parts):
        continue
    if src_file.name == "run_tests_databricks.py":
        continue
    ws_path = f"{workspace_dir}/{rel}"
    print(f"Uploading {rel}")
    upload_file(src_file, ws_path)

# Upload pyproject.toml
upload_file(repo_root / "pyproject.toml", f"{workspace_dir}/pyproject.toml")
print("Uploaded pyproject.toml")

# Create test notebook
test_notebook_content = """\
# Databricks notebook source
import subprocess, sys, os

# COMMAND ----------
os.chdir("/Workspace/insurance-lda-risk-tests")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "scikit-learn", "scipy", "numpy", "pandas", "matplotlib",
     "pytest", "pytest-cov", "-q", "--disable-pip-version-check"],
    capture_output=True, text=True
)
print("pip install done, rc=", result.returncode)

# COMMAND ----------
os.chdir("/Workspace/insurance-lda-risk-tests")
result2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".", "-q",
     "--disable-pip-version-check", "--no-build-isolation"],
    capture_output=True, text=True
)
print(result2.stdout[-2000:])
print(result2.stderr[-2000:])

# COMMAND ----------
os.chdir("/Workspace/insurance-lda-risk-tests")
result3 = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
    capture_output=True, text=True,
)
print(result3.stdout[-10000:])
if result3.stderr:
    print("STDERR:", result3.stderr[-2000:])
print("Return code:", result3.returncode)
assert result3.returncode == 0, "Tests failed"
"""

encoded_nb = base64.b64encode(test_notebook_content.encode()).decode()
w.workspace.import_(
    path=f"{workspace_dir}/run_tests",
    content=encoded_nb,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Uploaded test runner notebook")

# Submit job
job_run = w.jobs.submit(
    run_name="insurance-lda-risk tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{workspace_dir}/run_tests",
            ),
            new_cluster=compute.ClusterSpec(
                spark_version="15.4.x-scala2.12",
                node_type_id="m5d.large",
                num_workers=0,
                spark_conf={"spark.master": "local[*, 4]"},
                data_security_mode=compute.DataSecurityMode.SINGLE_USER,
            ),
        )
    ],
)
run_id = job_run.run_id
print(f"Submitted job run {run_id}")

# Poll for completion
while True:
    run = w.jobs.get_run(run_id=run_id)
    state = run.state.life_cycle_state.value
    print(f"  State: {state}")
    if state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        result_state = run.state.result_state.value if run.state.result_state else "UNKNOWN"
        print(f"  Result: {result_state}")
        try:
            output = w.jobs.get_run_output(run_id=run_id)
            if output.notebook_output and output.notebook_output.result:
                print("\n=== NOTEBOOK OUTPUT ===")
                print(output.notebook_output.result[:6000])
            if output.error:
                print(f"\nERROR: {output.error}")
        except Exception as e:
            print(f"Could not get output: {e}")
        break
    time.sleep(30)

print("Done.")
