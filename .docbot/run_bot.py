import os
import sys

# Ensure .docbot directory is in sys.path so we can import local scripts
# when running from the repository root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from update_summary import update_summary
from generate_docstrings import generate_docstrings
from google import genai


def generate_report(
    client: genai.Client, updated_files: list, old_summary: str, new_summary: str
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

List of files updated with new docstrings:
{files_list}

Instructions:
1. Generate a short, descriptive git commit message title (max 70 characters).
2. Generate a detailed Pull Request description (body) that summarizes:
   - Key changes in the architectural/mathematical summary (compare PREVIOUS vs UPDATED).
   - Which files were documented.
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


def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    print("--- Starting Documentation Bot ---")

    print("\n[Step 1/3] Updating Global Codebase Summary...")
    old_summary = ""
    new_summary = ""
    try:
        old_summary, new_summary = update_summary()
    except Exception as e:
        print(f"Error during summary update: {e}")

    print("\n[Step 2/3] Generating Docstrings for Changed Files...")
    updated_files = []
    try:
        updated_files = generate_docstrings()
    except Exception as e:
        print(f"Error during docstring generation: {e}")
        sys.exit(1)

    if updated_files or (old_summary != new_summary):
        print("\n[Step 3/3] Generating Change Report...")
        try:
            generate_report(client, updated_files, old_summary, new_summary)
        except Exception as e:
            print(f"Error during report generation: {e}")

    print("\n--- Documentation Bot Finished Successfully ---")


if __name__ == "__main__":
    main()
