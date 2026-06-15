import os
import sys

# Ensure .docbot directory is in sys.path so we can import local scripts
# when running from the repository root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from update_summary import update_summary
from generate_docstrings import generate_docstrings
from verify_and_fix import verify_and_fix
from google import genai
import subprocess


def run_tests():
    """Runs pytest and returns the return code and output."""
    print("Running tests...")
    result = subprocess.run(["uv", "run", "pytest"], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def generate_report(
    client: genai.Client,
    updated_files: list,
    old_summary: str,
    new_summary: str,
    tree_str: str,
    test_status: str,
):
    """Generate a commit message and PR body using Gemini."""
    files_list = "\n".join([f"- {f}" for f in updated_files])

    prompt = f"""
You are an expert technical coordinator. 
The EDGAR Documentation Bot has just finished updating the repository's documentation.

PREVIOUS Global Summary:
{old_summary}

UPDATED Global Summary:
{new_summary}

Current file structure:
{tree_str}

List of files updated with new docstrings:
{files_list}

Test Status Output:
{test_status}

Instructions:
1. Generate a short, descriptive git commit message title (max 70 characters).
2. Generate a detailed Pull Request description (body) that summarizes:
   - Key changes in the architectural/mathematical summary (compare PREVIOUS vs UPDATED).
   - Which files were documented.
   - The status of the tests (Passed or Failed). If they failed, briefly mention that manual review is needed.
3. Use Markdown for the PR description.
4. Output the result in this EXACT format:
---COMMIT_TITLE---
[Title here]
---PR_BODY---
[Body here]
"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    report_content = response.text
    with open(".docbot/report.md", "w") as f:
        f.write(report_content)
    print("Successfully generated .docbot/report.md")


def generate_folder_tree(startpath: str) -> str:
    """Generates a visual string of the folder tree structure, including .py files."""
    ignore_set = {"__pycache__", ".git", ".github", "docs", ".pytest_cache"}
    tree_lines = [f"{os.path.basename(startpath)}/"]

    def _build_tree(current_path: str, prefix: str = ""):
        try:
            # List items and sort them (directories first, then files)
            items = sorted(os.listdir(current_path))
            # Filter items: ignore hidden, ignore specified set, and only keep .py files or directories
            filtered_items = []
            for item in items:
                item_path = os.path.join(current_path, item)
                if item in ignore_set or item.startswith("."):
                    continue
                if os.path.isdir(item_path) or item.endswith(".py"):
                    filtered_items.append(item)

            for i, item in enumerate(filtered_items):
                item_path = os.path.join(current_path, item)
                is_last = i == len(filtered_items) - 1
                connector = "└── " if is_last else "├── "

                if os.path.isdir(item_path):
                    tree_lines.append(f"{prefix}{connector}{item}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_tree(item_path, new_prefix)
                else:
                    tree_lines.append(f"{prefix}{connector}{item}")
        except PermissionError:
            tree_lines.append(f"{prefix}└── [Permission Denied]")

    _build_tree(startpath)
    return "\n".join(tree_lines)


def main():
    tree_path = ".docbot/folder_tree.txt"
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    print("--- Starting Documentation Bot ---")

    print("\n[Step 1/3] Updating Global Codebase Summary...")
    old_summary = ""
    new_summary = ""
    try:
        tree_str = generate_folder_tree("edgar")
        with open(tree_path, "w") as f:
            f.write(tree_str)
        old_summary, new_summary = update_summary(tree_str)
    except Exception as e:
        print(f"Error during summary update: {e}")

    print("\n[Step 2/4] Generating Docstrings for Changed Files...")
    updated_files = []
    try:
        updated_files = generate_docstrings(tree_str)
    except Exception as e:
        print(f"Error during docstring generation: {e}")
        sys.exit(1)

    print("\n[Step 3/4] Verifying and Fixing Codebase...")
    try:
        verification_success = verify_and_fix()
        if not verification_success:
            print(
                "Warning: Verification and fixing failed. Proceeding with report anyway."
            )
    except Exception as e:
        print(f"Error during verification/fixing: {e}")

    print("\n[Step 4/5] Running Tests...")
    test_summary = "Tests were not run."
    try:
        ret_code, test_output = run_tests()
        if ret_code == 0:
            test_summary = "All tests passed successfully."
        else:
            test_summary = f"Some tests failed. Exit code: {ret_code}\n\nSelected Output:\n{test_output[-2000:]}"
    except Exception as e:
        test_summary = f"An error occurred while running tests: {e}"
    print(test_summary)

    if updated_files or (old_summary != new_summary):
        print("\n[Step 5/5] Generating Change Report...")
        try:
            generate_report(
                client, updated_files, old_summary, new_summary, tree_str, test_summary
            )
        except Exception as e:
            print(f"Error during report generation: {e}")

    print("\n--- Documentation Bot Finished Successfully ---")


if __name__ == "__main__":
    main()
