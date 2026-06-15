import os
from google import genai
from git_utils import get_last_doc_commit, get_diff, run_git_command


def update_summary(folder_tree: str = ""):
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    with open(".docbot/edgar_overview.md", "r") as f:
        overview = f.read()

    summary_path = ".docbot/code_summary.md"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            old_summary = f.read()
    else:
        old_summary = ""

    last_commit = get_last_doc_commit()
    if not last_commit:
        # Fallback: get the commit hash from 50 commits ago (or the first commit if less than 50)
        try:
            # --skip=50 finds the commit 50 steps back. If fewer than 50 exist, it returns nothing.
            last_commit = run_git_command(["log", "-1", "--skip=49", "--format=%H"])
            if not last_commit:
                # If less than 50 commits, get the very first one
                last_commit = run_git_command(["rev-list", "--max-parents=0", "HEAD"])
        except Exception:
            last_commit = None

    if last_commit:
        diff = get_diff(last_commit)
    else:
        diff = "Initial documentation run. No previous summary found."

    if not diff.strip() and old_summary.strip():
        print("No changes detected in edgar/. Skipping summary update.")
        return old_summary, old_summary

    prompt = f"""
You are an expert technical writer and scientific researcher. 
Your task is to update a global codebase summary file ('code_summary.md') for the EDGAR project.

EDGAR Overview:
{overview}

Repository Structure (edgar/ directory):
{folder_tree}

Current Summary:
{old_summary}

Recent Changes (Git Diff):
{diff}

Instructions:
1. Review the EDGAR Overview, the Repository Structure, and the Current Summary.
2. Analyze the Git Diff to understand what has changed in the codebase (specifically in the 'edgar/' directory).
3. Update the 'code_summary.md' to reflect these changes.
4. Maintain a high-level architectural and mathematical overview.
5. Be surgical: focus on what changed or was added.
6. Ensure scientific precision in describing algorithms, mathematical models, or data structures.
7. Return the ENTIRE updated 'code_summary.md' content.

Updated code_summary.md:
"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    new_summary = response.text

    with open(summary_path, "w") as f:
        f.write(new_summary)

    print("Successfully updated .docbot/code_summary.md")
    return old_summary, new_summary


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        exit(1)
    update_summary()
