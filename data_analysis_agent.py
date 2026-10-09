import csv
import io
import os
from pathlib import Path

from deepagents import SubAgent
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.middleware import (
    FilesystemMiddleware,
    SkillsMiddleware,
    SubAgentMiddleware,
    SummarizationMiddleware,
)
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langsmith.sandbox import SandboxClient

# 1. Initialize LangSmith Sandbox Backend
client = SandboxClient()
sandbox = client.create_sandbox(
    name="langchain-docs",
    snapshot_name="docs-test-ci",
)
backend = LangSmithSandbox(sandbox=sandbox)

# 2. Upload Sample Sales Data to Sandbox
rows = [
    ["Date", "Product", "Units", "Revenue"],
    ["2025-08-01", "Widget A", 10, 250],
    ["2025-08-02", "Widget B", 5, 125],
    ["2025-08-03", "Widget A", 7, 175],
    ["2025-08-04", "Widget C", 3, 90],
]

buf = io.StringIO()
csv.writer(buf).writerows(rows)
backend.upload_files([("/sales.csv", buf.getvalue().encode())])

# 3. Create and Upload Skills Directory
skills_dir = (Path(__file__).resolve().parent / "skills").resolve()

# Create local skill directory if it doesn't exist
skill_file_path = skills_dir / "pandas-patterns" / "SKILL.md"
skill_file_path.parent.mkdir(parents=True, exist_ok=True)
skill_file_path.write_text(
    """---
name: pandas-patterns
description: Common pandas and matplotlib patterns for data analysis and visualization
---

## Data loading
Use `pd.read_csv()` for CSV files. Always check `df.info()` and `df.describe()` first.

## Visualization
Use `matplotlib` for bar charts, `seaborn` for statistical plots. Save figures with `plt.savefig("output.png", dpi=150, bbox_inches="tight")`.

## Reporting
Write a markdown summary to `report.md` alongside any generated charts.
"""
)

# Upload skills to sandbox filesystem
skill_files: list[tuple[str, bytes]] = []
for path in sorted(skills_dir.rglob("*")):
    if not path.is_file():
        continue
    rel = path.resolve().relative_to(skills_dir)
    skill_files.append((f"/skills/{rel.as_posix()}", path.read_bytes()))

backend.upload_files(skill_files)

# 4. Define Subagent Configuration
model = "google_genai:gemini-3.6-flash"

visualizer: SubAgent = {
    "name": "visualizer",
    "description": "Generates charts and visualizations from data files in the sandbox.",
    "system_prompt": (
        "You are a data visualization specialist. Write Python scripts using matplotlib "
        "and seaborn. Save all figures as PNG files."
    ),
    "tools": [],
    "model": model,
}

# 5. Assemble Agent with Middleware Harness
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        SkillsMiddleware(backend=backend, sources=["/skills/"]),
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[visualizer]),
    ],
)

# 6. Stream Agent Execution
if __name__ == "__main__":
    upload_stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze /sales.csv using our pandas patterns, then "
                        "create a bar chart of revenue by product."
                    ),
                }
            ]
        },
        version="v3",
        config={"recursion_limit": 15},
    )

    for item in upload_stream.messages:
        print(item.text)