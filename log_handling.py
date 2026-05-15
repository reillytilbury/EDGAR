import asyncio
import html
import json
import os
import re
from typing import Union

from dotenv import load_dotenv
from google import genai

from utils import call_llm_async


_EQUATION_PROMPT = """\
Below is the docstring of a neuron model. It contains a description of the model and usually an equation.

Extract the primary mathematical equation for the model's firing rate and list every free parameter used by \
the equation. Return only the free-parameter list and the LaTeX equation. Do not include any explanation, \
commentary, or extra text.

Use exactly this output format:
FREE PARAMS:
$param_1$, $param_2$, $param_3$.

EQUATION:
$$...$$

Follow these formatting rules strictly:
- Use concise mathematical symbols for all variables — never spell out words. \
For example, use $b$ not $\\text{{baseline}}$, $A$ not $\\text{{amplitude}}$, $\\sigma$ not $\\text{{tuning\\_width}}$.
- Circular distance between angles should be written as $|\\theta - \\theta'|$, never as a named function like $d(\\theta, \\theta')$ or $\\text{{circ\\_dist}}(\\cdot)$.
- Never use * for multiplication. Use juxtaposition or \\cdot where needed.
- Write fractions using \\frac{{numerator}}{{denominator}}, not a/b.
- Write the full equation in one continuous expression — do not introduce auxiliary functions. \
For example, do not write $R(\\theta) = b + A_1 G(\\theta, \\theta_1) + A_2 G(\\theta, \\theta_2)$ and then define $G$ separately; \
instead expand everything inline.
- When subscripts contain words, use \\text{{}} for neat rendering. \
For example, write $\\theta_{{\\text{{pref}}}}$ not $\\theta_{{pref}}$.
- If a peak has a hard side-dependent width, such as \\sigma = \\sigma_{{\\text{{left}}}} on one side of the peak \
and \\sigma = \\sigma_{{\\text{{right}}}} on the other side, represent that paired width parameter as \
$\\sigma_{{\\pm}}$. If there is more than one such theta-dependent paired width, use $\\sigma_{{i\\pm}}$. \
Apply the same convention to equivalent hard side-dependent parametrizations such as \
$\\sigma = \\sigma_0(1 + k\\operatorname{{sign}}(\\theta - \\theta_0))$. Do not apply this convention to \
smooth theta-dependent width functions.
- In FREE PARAMS, include only symbols that are free parameters of the model, not input variables, constants, \
intermediate functions, or the response variable.
- In FREE PARAMS, format each symbol as an inline LaTeX math expression using $...$, separate symbols with \
commas, and end the list with a period.

Example docstring:
\"\"\"
Two-component Gaussian orientation tuning model. The firing rate is the sum of a baseline and two Gaussian \
bumps with amplitudes A_1 and A_2. The bumps share preferred orientation theta_pref and width sigma, and the \
second bump is centered at the opposite direction theta_pref + pi. For stimulus angle theta, the response is:
r(theta) = b + A_1 exp(-|theta - theta_pref|^2 / (2 sigma^2)) \
+ A_2 exp(-|theta - (theta_pref + pi)|^2 / (2 sigma^2)).
\"\"\"

Example output:
FREE PARAMS:
$b$, $A_1$, $A_2$, $\\theta_{{\\text{{pref}}}}$, $\\sigma$.

EQUATION:
$$r(\\theta) = b + A_1 \\exp\\left(-\\frac{{|\\theta - \\theta_{{\\text{{pref}}}}|^2}}{{2\\sigma^2}}\\right) + A_2 \\exp\\left(-\\frac{{|\\theta - (\\theta_{{\\text{{pref}}}} + \\pi)|^2}}{{2\\sigma^2}}\\right)$$

Docstring:
{docstring}
"""


_NORMALIZE_EQUATIONS_PROMPT = """\
Below is a batch of neuron-model equation rows from a Markdown file.

Standardize the notation across all rows in this batch so equivalent parameters use the same symbols. \
For example, use one consistent symbol for baseline terms, one consistent symbol for exponents, and one \
consistent style for amplitudes, preferred angles, widths, offsets, mixture weights, and other repeated \
parameter roles. Prefer concise notation such as b for baseline, A_i for amplitudes, \
\\theta_{{\\text{{pref}},i}} for preferred angles, \\sigma_i for widths, and p for exponents unless a \
different symbol is clearly more appropriate across the batch.

Preserve the mathematical meaning of each row. Do not add or remove model components. Do not change Iter, \
Island, or Batch. Update Free Params so it exactly matches the free parameters used in Equation. Include \
only free parameters, not input variables, constants, intermediate functions, or the response variable.

Apply these formatting rules:
- Return only a JSON array, with no Markdown, code fences, comments, or explanation.
- The first character of your response must be [ and the last character must be ].
- This must be valid JSON: every LaTeX backslash inside a JSON string must be escaped as \\.
- Return the same number of rows in the same order.
- Each row must have exactly these keys: "Iter", "Island", "Batch", "Free Params", "Equation".
- "Free Params" must be a comma-separated list of inline LaTeX math expressions ending with a period, \
for example: "$b$, $A_1$, $\\theta_{{\\text{{pref}}}}$, $\\sigma$."
- "Equation" must be a LaTeX display equation wrapped in double dollar signs.
- Use concise mathematical symbols for all variables and parameters.
- Use \\text{{}} when subscripts contain words.
- Never use * for multiplication. Use juxtaposition or \\cdot where needed.
- Write fractions using \\frac{{numerator}}{{denominator}}, not a/b.
- Keep equations inline as a single expression; do not define auxiliary functions separately.
- If a peak has a hard side-dependent width, use \\sigma_{{\\pm}}. If there is more than one such \
theta-dependent paired width, use \\sigma_{{i\\pm}}. Apply this to hard sign-based parametrizations too, \
but not to smooth theta-dependent width functions.

Rows:
{rows_json}
"""


_FIX_NORMALIZED_JSON_PROMPT = """\
The previous response was intended to be a JSON array of normalized equation rows, but it could not be parsed.

JSON parser error:
{error_message}

Fix only the JSON syntax. Preserve the exact same row objects, values, row order, and mathematical content. \
Do not change notation further. Do not add or remove rows. The corrected response must be valid JSON.

Rules:
- Return only the JSON array, with no Markdown, code fences, comments, or explanation.
- The first character of your response must be [ and the last character must be ].
- Every LaTeX backslash inside a JSON string must be escaped as \\.
- Each row must have exactly these keys: "Iter", "Island", "Batch", "Free Params", "Equation".
- "Free Params" values must be inline LaTeX math expressions, such as "$b$, $A_1$, $\\sigma$."

Original source rows, for checking row identity:
{rows_json}

Invalid response to correct:
{response_text}
"""


def get_child_docstring(log_path: str, iter: int, island: int, batch: int) -> str:
    """Return the docstring of the child neuron_model generated at (iter, island, batch)."""
    with open(log_path, "r") as f:
        text = f.read()

    id_pattern = re.compile(
        rf"id={re.escape(str(iter))},{re.escape(str(island))},{re.escape(str(batch))}\n",
    )
    id_match = id_pattern.search(text)
    if not id_match:
        raise ValueError(f"No block found for id={iter},{island},{batch}")

    # Slice to the next id= block (or EOF) so we don't bleed into a later entry
    next_id = re.search(r"\nid=\d+,\d+,\d+\n", text[id_match.end():])
    block_end = id_match.end() + next_id.start() if next_id else len(text)
    block = text[id_match.start():block_end]

    # Find "Neuron Model: \n" (skip the JAX variant — it comes later and has no docstring)
    nm_match = re.search(r"\nNeuron Model: \n", block)
    if not nm_match:
        raise ValueError(f"No 'Neuron Model:' section found for id={iter},{island},{batch}")

    child_code = block[nm_match.end():]

    # Extract the first triple-quoted docstring
    ds_match = re.search(r'"""(.*?)"""', child_code, re.DOTALL)
    if not ds_match:
        raise ValueError(f"No docstring found in child model for id={iter},{island},{batch}")

    return ds_match.group(1).strip()


async def docstring_to_latex_equation(
    docstring: str,
    client: genai.Client,
) -> Union[str, None]:
    """Return free parameters and a LaTeX math-mode equation extracted from the docstring.

    Uses gemini-2.5-flash with thinking disabled (budget=0).
    Designed to be called concurrently via asyncio.gather or similar.
    """
    prompt = _EQUATION_PROMPT.format(docstring=docstring)
    return await call_llm_async(
        prompt_text=prompt,
        client=client,
        model_name="gemini-2.5-flash",
        temperature=0.0,
        thinking_budget=0,
    )


def parse_equation_response(response: Union[str, None]) -> tuple[str, str]:
    """Split an LLM response into free parameters and equation text."""
    if not response:
        raise ValueError("Empty equation response")

    response = response.strip()
    free_params_match = re.search(
        r"FREE PARAMS:\s*(.*?)\s*EQUATION:",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    equation_match = re.search(
        r"EQUATION:\s*(.*?)\s*$",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not free_params_match or not equation_match:
        raise ValueError("Equation response must contain FREE PARAMS and EQUATION sections")

    free_params = free_params_match.group(1).strip()
    equation = equation_match.group(1).strip()
    if not free_params or not equation:
        raise ValueError("Equation response has an empty FREE PARAMS or EQUATION section")
    return free_params, equation


def html_table_cell(text: str) -> str:
    """Return text that is safe to place inside an HTML table cell."""
    return html.escape(" ".join(text.split()), quote=False)


_ROW_KEYS = ("Iter", "Island", "Batch", "Free Params", "Equation")


async def gather_with_progress(label: str, coroutines: list) -> list:
    """Run coroutines concurrently while printing completion progress."""
    total = len(coroutines)
    if total == 0:
        return []

    async def indexed(index: int, coroutine):
        return index, await coroutine

    print(f"{label}: 0/{total}", flush=True)
    tasks = [
        asyncio.create_task(indexed(index, coroutine))
        for index, coroutine in enumerate(coroutines)
    ]
    results = [None] * total
    for complete, task in enumerate(asyncio.as_completed(tasks), start=1):
        index, result = await task
        results[index] = result
        print(f"{label}: {complete}/{total}", flush=True)
    return results


def read_equation_rows(markdown_path: str) -> list[dict[str, str]]:
    """Read equation rows from this module's HTML-table Markdown format."""
    print(f"Reading equation rows from {markdown_path}...", flush=True)
    with open(markdown_path, "r") as f:
        text = f.read()
    rows = []
    for raw_row in re.findall(r"<tr>(.*?)</tr>", text, flags=re.DOTALL):
        cells = re.findall(r"<td>(.*?)</td>", raw_row, flags=re.DOTALL)
        if not cells:
            continue
        if len(cells) != len(_ROW_KEYS):
            raise ValueError(f"Expected {len(_ROW_KEYS)} cells, found {len(cells)}")
        rows.append({
            key: html.unescape(cell.strip())
            for key, cell in zip(_ROW_KEYS, cells)
        })
    print(f"Loaded {len(rows)} equation row(s).", flush=True)
    return rows


def write_equation_rows(markdown_path: str, rows: list[dict[str, str]]) -> None:
    """Write equation rows to this module's HTML-table Markdown format."""
    print(f"Writing {len(rows)} equation row(s) to {markdown_path}...", flush=True)
    with open(markdown_path, "w") as f:
        f.write("# Neuron Model Equations\n\n")
        f.write("<table>\n")
        f.write("  <thead>\n")
        f.write("    <tr><th>Iter</th><th>Island</th><th>Batch</th><th>Free Params</th><th>Equation</th></tr>\n")
        f.write("  </thead>\n")
        f.write("  <tbody>\n")
        for row in rows:
            f.write("    <tr>")
            for key in _ROW_KEYS:
                f.write(f"<td>{html_table_cell(str(row[key]))}</td>")
            f.write("</tr>\n")
        f.write("  </tbody>\n")
        f.write("</table>\n")
    print(f"Finished writing {markdown_path}.", flush=True)


def validate_normalized_rows(
    original_rows: list[dict[str, str]],
    normalized_rows: object,
) -> Union[list[dict[str, str]], None]:
    """Return normalized rows if the LLM preserved row shape and identity."""
    if not isinstance(normalized_rows, list) or len(normalized_rows) != len(original_rows):
        print(
            "Normalized equation batch had the wrong row count/type: "
            f"expected list[{len(original_rows)}], got {type(normalized_rows).__name__}.",
            flush=True,
        )
        return None

    rows = []
    for index, (original, normalized) in enumerate(zip(original_rows, normalized_rows)):
        if not isinstance(normalized, dict):
            print(
                f"Normalized row {index} had type {type(normalized).__name__}, expected dict.",
                flush=True,
            )
            return None
        if set(normalized) != set(_ROW_KEYS):
            print(
                f"Normalized row {index} had keys {sorted(normalized)}, "
                f"expected {list(_ROW_KEYS)}.",
                flush=True,
            )
            return None
        if any(str(normalized[key]) != str(original[key]) for key in _ROW_KEYS[:3]):
            print(
                f"Normalized row {index} changed row identity: "
                f"expected {[original[key] for key in _ROW_KEYS[:3]]}, "
                f"got {[normalized[key] for key in _ROW_KEYS[:3]]}.",
                flush=True,
            )
            return None
        rows.append({key: str(normalized[key]).strip() for key in _ROW_KEYS})
    return rows


def parse_json_response(response: str) -> object:
    """Parse JSON, allowing accidental Markdown code fences."""
    response = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        response = fenced.group(1).strip()
    return json.loads(response)


async def correct_normalization_json(
    rows: list[dict[str, str]],
    response_text: str,
    error_message: str,
    client: genai.Client,
    model_name: str,
) -> Union[object, None]:
    """Ask the model to repair invalid JSON from a normalization response."""
    print("Asking model to correct invalid normalization JSON...", flush=True)
    prompt = _FIX_NORMALIZED_JSON_PROMPT.format(
        error_message=error_message,
        rows_json=json.dumps(rows, indent=2),
        response_text=response_text,
    )
    corrected = await call_llm_async(
        prompt_text=prompt,
        client=client,
        model_name=model_name,
        temperature=0.0,
        thinking_budget=1,
    )
    if corrected is None:
        print("JSON correction model returned no response.", flush=True)
        return None

    corrected_text = corrected.strip()
    try:
        return parse_json_response(corrected_text)
    except json.JSONDecodeError as exc:
        print(f"Could not parse corrected normalization JSON: {exc}", flush=True)
        print("Raw corrected response follows:", flush=True)
        print(corrected_text if corrected_text else "<empty response>", flush=True)
        print("End raw corrected response.", flush=True)
        return None


async def normalize_equation_batch(
    rows: list[dict[str, str]],
    client: genai.Client,
    model_name: str = "gemini-2.5-pro",
) -> Union[list[dict[str, str]], None]:
    """Ask Gemini Pro to standardize notation for one batch of equation rows."""
    prompt = _NORMALIZE_EQUATIONS_PROMPT.format(
        rows_json=json.dumps(rows, indent=2),
    )
    response = await call_llm_async(
        prompt_text=prompt,
        client=client,
        model_name=model_name,
        temperature=0.0,
        thinking_budget=1,
    )
    if response is None:
        print("Normalization model returned no response.", flush=True)
        return None

    response_text = response.strip()
    try:
        parsed = parse_json_response(response_text)
    except json.JSONDecodeError as exc:
        print(f"Could not parse normalized equation batch as JSON: {exc}", flush=True)
        print("Raw normalization response follows:", flush=True)
        print(response_text if response_text else "<empty response>", flush=True)
        print("End raw normalization response.", flush=True)
        parsed = await correct_normalization_json(
            rows=rows,
            response_text=response_text,
            error_message=str(exc),
            client=client,
            model_name=model_name,
        )
        if parsed is None:
            return None

    normalized = validate_normalized_rows(rows, parsed)
    if normalized is None:
        print("Raw parsed normalization response follows:", flush=True)
        print(json.dumps(parsed, indent=2), flush=True)
        print("End raw parsed normalization response.", flush=True)
    return normalized


async def normalize_equation_rows_once(
    rows: list[dict[str, str]],
    batch_size: int,
    client: genai.Client,
    model_name: str = "gemini-2.5-pro",
) -> list[dict[str, str]]:
    """Normalize all rows once, split into concurrent batches of batch_size."""
    batches = [
        rows[start:start + batch_size]
        for start in range(0, len(rows), batch_size)
    ]
    normalized_batches = list(await gather_with_progress(
        f"k={batch_size} normalization batches",
        [
            normalize_equation_batch(batch, client, model_name=model_name)
            for batch in batches
        ],
    ))
    failed = sum(batch is None for batch in normalized_batches)
    if failed:
        print(
            f"Keeping original rows for {failed} failed normalization batch(es) at k={batch_size}.",
            flush=True,
        )

    return [
        row
        for original_batch, normalized_batch in zip(batches, normalized_batches)
        for row in (normalized_batch if normalized_batch is not None else original_batch)
    ]


async def normalize_equations_markdown(
    markdown_path: str,
    client: genai.Client,
    initial_batch_size: int = 10,
    model_name: str = "gemini-2.5-pro",
) -> None:
    """Iteratively standardize equation notation in an HTML-table Markdown file."""
    rows = read_equation_rows(markdown_path)
    if not rows:
        raise ValueError(f"No equation rows found in {markdown_path}")

    batch_size = initial_batch_size
    while True:
        print(f"Normalizing {len(rows)} equation row(s) with k={batch_size}...")
        rows = await normalize_equation_rows_once(
            rows,
            batch_size=batch_size,
            client=client,
            model_name=model_name,
        )
        write_equation_rows(markdown_path, rows)
        print(f"Completed normalization pass k={batch_size}.", flush=True)
        if batch_size >= len(rows):
            break
        batch_size *= 2
    print("Finished all normalization passes.", flush=True)


async def generate_equations_markdown(
    log_path: str,
    output_path: str,
    client: genai.Client,
    n_iters: int = 9,
    n_islands: int = 8,
    n_batches: int = 6,
) -> None:
    """Extract child docstrings for every (iter, island, batch), fetch free
    parameters and LaTeX equations, and write a Markdown file to output_path.

    Rows are ordered chronologically: iter → island → batch.
    Skipped/missing entries (LLM failures at generation time) are omitted silently.
    Failed equation requests are omitted from the Markdown output.
    """
    triples = [
        (it, isl, b)
        for it in range(n_iters)
        for isl in range(n_islands)
        for b in range(n_batches)
    ]

    # Collect docstrings synchronously (pure string parsing — fast)
    print(f"Collecting docstrings from {len(triples)} possible row(s): 0/{len(triples)}", flush=True)
    rows = []
    for index, (it, isl, b) in enumerate(triples, start=1):
        try:
            docstring = get_child_docstring(log_path, it, isl, b)
        except ValueError:
            pass  # skipped / failed generation
        else:
            rows.append((it, isl, b, docstring))
        if index % 25 == 0 or index == len(triples):
            print(
                f"Collecting docstrings: {index}/{len(triples)} checked, {len(rows)} found",
                flush=True,
            )
    print(f"Collected {len(rows)} docstring row(s).", flush=True)

    # Fire off all equation requests concurrently. Failed requests are omitted.
    equations = list(await gather_with_progress(
        "Extracting equations",
        [
            docstring_to_latex_equation(docstring, client)
            for *_, docstring in rows
        ],
    ))

    equation_rows = []
    for (it, isl, b, _), response in zip(rows, equations):
        if response is None:
            print(f"Dropping row iter={it}, island={isl}, batch={b}: equation extraction failed.", flush=True)
            continue
        try:
            free_params, equation = parse_equation_response(response)
        except ValueError as exc:
            print(f"Dropping row iter={it}, island={isl}, batch={b}: {exc}", flush=True)
            continue
        equation_rows.append({
            "Iter": str(it),
            "Island": str(isl),
            "Batch": str(b),
            "Free Params": free_params,
            "Equation": equation,
        })
    write_equation_rows(output_path, equation_rows)
    print(f"Generated {len(equation_rows)} equation row(s).", flush=True)


if __name__ == "__main__":
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    async def main() -> None:
        output_path = "equations.md"
        print("Starting equation extraction.", flush=True)
        await generate_equations_markdown(
            log_path="program_databases/05-07/21-01-01 (GT1)/hypothesis_engine.log",
            output_path=output_path,
            client=client,
        )
        print("Starting equation notation normalization.", flush=True)
        await normalize_equations_markdown(output_path, client)
        print("Done.", flush=True)

    asyncio.run(main())
