import subprocess
import os
import re
import sys
from google import genai


def run_commit_check():
    """Runs make commit-check and returns the return code and output."""
    result = subprocess.run(["make", "commit-check"], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def fix_file(client: genai.Client, file_path: str, error_output: str) -> bool:
    """Uses Gemini to fix linting/formatting errors in a file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return False

    with open(file_path, "r") as f:
        content = f.read()

    prompt = f"""
You are an expert Python developer. 
The following file has linting or formatting errors after a documentation update.
Your task is to fix these errors while preserving the docstrings and functional logic.

File Path: {file_path}
Error Output:
{error_output}

Original File Content:
{content}

Instructions:
1. Fix the errors described in the Error Output.
2. Ensure the code is syntactically correct and follows Python best practices.
3. Do NOT remove docstrings unless they are the source of the error (e.g., malformed).
4. **CRITICAL**: Return ONLY the raw Python code. Do NOT enclose the code in markdown backticks like ```python ... ``` or ```. 
5. Start your response directly with the first line of the Python file (e.g., an import, a docstring, or a comment).

Updated File Content:
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        updated_content = response.text

        if updated_content:
            # Manually strip markdown code blocks if the LLM ignored instructions
            updated_content = updated_content.strip()
            if updated_content.startswith("```python"):
                updated_content = updated_content[9:]
            elif updated_content.startswith("```"):
                updated_content = updated_content[3:]

            if updated_content.endswith("```"):
                updated_content = updated_content[:-3]

            updated_content = updated_content.strip()

            # Simple check to avoid overwriting with garbage
            if (
                "def " in updated_content
                or "class " in updated_content
                or "import " in updated_content
                or updated_content.startswith('"""')
            ):
                with open(file_path, "w") as f:
                    f.write(updated_content)
                print(f"Successfully applied fixes to {file_path}")
                return True
            else:
                print(
                    f"Warning: Gemini returned suspicious content for {file_path}. Skipping update."
                )
        return False
    except Exception as e:
        print(f"Error during Gemini call for {file_path}: {e}")
        return False


def verify_and_fix(max_retries: int = 3) -> bool:
    """Orchestrates the verification and fix loop."""
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return False

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    for i in range(max_retries):
        print(f"\n--- Running commit-check (Attempt {i + 1}/{max_retries}) ---")
        returncode, output = run_commit_check()

        if returncode == 0:
            print("Verification passed!")
            return True

        print(f"Verification failed. Output:\n{output}")

        # Extract file paths from ruff/make output
        # Matches paths ending in .py followed by a colon
        file_paths = set(re.findall(r"([\w/.-]+\.py):", output))

        if not file_paths:
            print("No problematic files identified in output.")
            break

        for file_path in file_paths:
            print(f"Attempting to fix {file_path}...")
            fix_file(client, file_path, output)

    # Final check after all attempts
    print("\n--- Final verification check ---")
    returncode, output = run_commit_check()
    if returncode == 0:
        print("Verification passed after fixes!")
        return True
    else:
        print(f"Verification still failing. Final output:\n{output}")
        return False


if __name__ == "__main__":
    if verify_and_fix():
        sys.exit(0)
    else:
        print("Failed to fix all issues after multiple attempts.")
        sys.exit(1)
