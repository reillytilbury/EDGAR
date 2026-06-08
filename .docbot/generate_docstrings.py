import os
import glob
from google import genai
from google.genai import types
from git_utils import get_last_doc_commit, get_changed_files


def read_repository_file(path: str) -> str:
    """Reads and returns the content of a file in the repository.
    Use this to understand context, class definitions, or function signatures in other modules.
    """
    # Security: only allow reading files within the repo
    if ".." in path or path.startswith("/"):
        return "Error: Access denied. Paths must be relative to the repository root."

    if not os.path.exists(path):
        return f"Error: File '{path}' not found."

    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def generate_docstrings(folder_tree: str = ""):
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
        print(f"Processing {file_path} (Agentic mode)...")
        with open(file_path, "r") as f:
            content = f.read()

        prompt = f"""
You are an expert Python scientific software engineer and technical writer. 
Your task is to add or update Google-style docstrings in a Python source file for the EDGAR project.

EDGAR Overview:
{overview}

Repository Structure (edgar/ directory):
{folder_tree}

Codebase Summary:
{summary}

Documentation Guidelines:
{guidelines}

Target File Content at {file_path}:
{content}

Instructions:
1. Read the provided file content and understand its functionality, paying attention to its location in the broader Repository Structure and the Codebase Summary.
2. If you need to see the definition of an imported class, a base class, or a utility function from another file to write accurate docstrings, use the `read_repository_file` tool.
3. Add or update docstrings, paying attention to the context of the entire codebase and the specific role of this file within it, and following the Documentation Guidelines.
4. **Important:** Only modify existings docstrings if they do not match the Documentation Guidelines or do not match the content of the function/module. 
5. Use the provided context to ensure mathematical and scientific precision.
6. **CRITICAL**: Do NOT change any functional code, imports, or logic. Only modify docstrings.
7. Return the ENTIRE updated file content, do not include ```python ... ``` backticks to enclose the code.

Updated File Content:
"""

        # Using tools for agentic exploration with strict limits
        config = types.GenerateContentConfig(
            max_output_tokens=20000,
            tools=[read_repository_file],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=5,  # Limit Gemini to 5 context-seeking turns per file
            ),
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt, config=config
            )

            updated_content = response.text

            if updated_content is None:
                print(f"Warning: Gemini returned no text for {file_path}.")
                # Check for tool calls or finish reason
                if response.candidates and response.candidates[0].content.parts:
                    parts = response.candidates[0].content.parts
                    if any(p.function_call for p in parts):
                        print(
                            "  Note: Response ended on a function call. Possibly hit maximum_remote_calls limit."
                        )
                    print(f"  Finish reason: {response.candidates[0].finish_reason}")
                continue

            # Simple safety check: ensure the LLM didn't just return garbage
            # Relaxed for __init__.py which might be empty or just docstrings
            is_init = file_path.endswith("__init__.py")
            is_valid = (
                "def " in updated_content
                or "class " in updated_content
                or "import " in updated_content
                or is_init
            )

            if is_valid:
                with open(file_path, "w") as f:
                    f.write(updated_content)
                print(f"Successfully updated {file_path}")
                updated_files.append(file_path)
            else:
                print(
                    f"Warning: Gemini returned suspicious content for {file_path}. Skipping update."
                )
                print(
                    f"--- Content Start ---\n{updated_content[:200]}...\n--- Content End ---"
                )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return updated_files


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        exit(1)
    generate_docstrings()
