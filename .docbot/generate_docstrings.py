import os
import glob
from google import genai
from git_utils import get_last_doc_commit, get_changed_files


def generate_docstrings():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    with open(".docbot/documentation_guidelines.md", "r") as f:
        guidelines = f.read()

    with open(".docbot/edgar_overview.md", "r") as f:
        overview = f.read()

    summary_path = ".docbot/code_summary.md"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = f.read()
    else:
        summary = "No global summary available."

    last_commit = get_last_doc_commit()
    if last_commit:
        changed_files = get_changed_files(last_commit)
    else:
        # If no last commit found, process all .py files in edgar/
        changed_files = glob.glob("edgar/**/*.py", recursive=True)

    if not changed_files:
        print("No changed files detected. Skipping docstring generation.")
        return []

    updated_files = []
    for file_path in changed_files:
        print(f"Processing {file_path}...")
        with open(file_path, "r") as f:
            content = f.read()

        prompt = f"""
You are an expert Python developer and technical writer. 
Your task is to add or update Google-style docstrings in a Python source file for the EDGAR project.

EDGAR Overview:
{overview}

Codebase Summary:
{summary}

Documentation Guidelines:
{guidelines}

Target File Content:
```python
{content}
```

Instructions:
1. Read the provided file content.
2. Insert or update a module-level docstring at the very top of the file (before any imports).
3. The module docstring must include:
   - A brief description of the module's purpose.
   - A 'Usage Example:' section with a realistic code snippet.
4. Add or update docstrings for all classes and functions following Google Style (Args, Returns, Raises).
5. Use the EDGAR Overview and Codebase Summary to ensure mathematical and scientific precision in the docstrings.
6. **CRITICAL**: Do NOT change any functional code, imports, or logic. Only modify docstrings.
7. Return the ENTIRE updated file content.

Updated File Content:
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",  # or gemini-2.5-flash if available
            contents=prompt,
        )

        updated_content = response.text

        # Simple safety check: ensure the LLM didn't just return garbage
        if (
            "def " in updated_content
            or "class " in updated_content
            or "import " in updated_content
        ):
            with open(file_path, "w") as f:
                f.write(updated_content)
            print(f"Successfully updated {file_path}")
            updated_files.append(file_path)
        else:
            print(
                f"Warning: Gemini returned suspicious content for {file_path}. Skipping update."
            )

    return updated_files


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        exit(1)
    generate_docstrings()
