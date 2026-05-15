# Converting equations.md to Publication-Ready HTML

## Overview

This guide provides step-by-step instructions for converting an `equations.md` file (HTML table format containing mathematical equations) into a beautiful, publication-ready HTML document suitable for scientific papers.

## Input Format

The input `equations.md` file should be an HTML table with the following structure:

```html
<table>
  <thead>
    <tr><th>Iter</th><th>Island</th><th>Batch</th><th>Free Params</th><th>Equation</th></tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0</td><td>0</td><td>$b$, $A_1$, ...</td><td>$$R(\theta) = ...$$</td></tr>
    ...
  </tbody>
</table>
```

**Key Points:**
- Iter, Island, Batch columns define the appearance index (tuple ordering)
- Free Params column contains LaTeX expressions for parameters
- Equation column contains full LaTeX display equations
- Rows may have arbitrary (iter, island, batch) combinations (not necessarily sequential)

## Output Format

The generated HTML has:
- **Program** column: Sequential row numbers (1, 2, 3, ...) instead of equation IDs
- **Free Parameters** column: LaTeX formatted parameters
- **Loss** column: (optional) Loss values if available from log file
- **Equation** column: Full, clean mathematical equations

**Features:**
- Professional scientific styling (Times New Roman, black & white)
- MathJax v3 rendering for LaTeX
- Sorted by (iter, island, batch) tuple
- All formatting quirks removed

## Step-by-Step Processing

### Step 1: Parse Input File

Extract rows from the HTML table:
- Use regex to find all `<tr><td>...</td>...<td>...</td></tr>` patterns
- Extract: iter (int), island (int), batch (int), free_params (str), equation (str)
- Store as list of dictionaries

**Regex Pattern:**
```
<tr><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td><td>(.*?)</td><td>(.*?)</td></tr>
```

### Step 2: Optional - Extract Losses from Log

If you have access to `hypothesis_engine.log`:

```python
# Find blocks marked with: id=iter,island,batch
# Then search within that block for: Loss: XX.XX
# Build map: {(iter, island, batch): loss_value}
```

This is optional. If no log file, proceed without loss values.

### Step 3: Sort by Appearance Index

Sort all rows by their (iter, island, batch) tuple using dictionary/tuple ordering:
```python
rows.sort(key=lambda x: (x['iter'], x['island'], x['batch']))
```

This ensures equations are in chronological order of generation.

### Step 4: Fix Formatting Issues

#### Issue 4a: Complex atan2 Expressions

**Problem:** Some equations contain:
```latex
\operatorname{atan2}(\sin(\theta - \theta_{\text{pref}}), \cos(\theta - \theta_{\text{pref}}))
```

**Solution:** Replace with simple absolute value:
```latex
|\theta - \theta_{\text{pref}}|
```

**Implementation:**
- Use regex to find pattern: `\operatorname{atan2}(\sin(X), \cos(X))`
- Replace with: `|X|`
- Repeat until no more matches (handles multiple occurrences in same equation)

#### Issue 4b: circ_dist Expressions

**Problem:** Some equations use:
```latex
\operatorname{circ\_dist}(\theta, \theta_{\text{pref}} + \pi + \phi)
```

**Solution:** Replace with difference notation:
```latex
|\theta - (\theta_{\text{pref}} + \pi + \phi)|
```

**Implementation:**
- Find pattern: `\operatorname{circ\_dist}(ARG1, ARG2)`
- Replace with: `|ARG1 - (ARG2)|`

#### Issue 4c: Double Bars

**Problem:** After previous replacements, double bars may appear:
```latex
||\theta - \theta_{\text{pref}}||
```

**Solution:** Simple string replacement:
```python
equation = equation.replace("||", "|")
```

### Step 5: Generate HTML

Create HTML with:

1. **Header Section:**
   - Title: "Discovered Equations and Free Parameters"
   - MathJax configuration to render `$...$` (inline) and `$$...$$` (display)

2. **MathJax Configuration:**
   ```javascript
   MathJax = {
       tex: {
           inlineMath: [['$', '$'], ['\\(', '\\)']],
           displayMath: [['$$', '$$'], ['\\[', '\\]']]
       }
   };
   ```
   **Why:** equations.md uses `$...$` for inline math but MathJax v3 doesn't recognize it by default.

3. **Styling:**
   - Font: Times New Roman (standard for scientific papers)
   - Colors: Black & white only (professional)
   - Borders: Top/bottom lines only (like scientific tables)
   - No background colors or hover effects

4. **Table Structure:**
   - Columns: Program | Free Parameters | [Loss] | Equation
   - Row numbering: 1, 2, 3, ... (sequential)
   - Last row has double bottom border

### Step 6: Write Output

Save as `.html` file. The file is self-contained and can be opened in any web browser.

## Usage

### Basic Usage (No Losses)

```python
from equations_to_html import EquationsToHTML

converter = EquationsToHTML(
    equations_md_path="equations.md",
    output_html_path="discovered_equations.html"
)
converter.convert()
```

### With Loss Values

```python
converter = EquationsToHTML(
    equations_md_path="equations.md",
    output_html_path="discovered_equations.html",
    log_path="hypothesis_engine.log"  # Path to log with Loss values
)
converter.convert()
```

## Implementation Checklist

- [ ] Parse HTML table from equations.md
- [ ] Extract (iter, island, batch, free_params, equation) tuples
- [ ] (Optional) Extract losses from log file
- [ ] Sort rows by (iter, island, batch)
- [ ] Fix atan2 expressions
- [ ] Fix circ_dist expressions
- [ ] Fix double bars
- [ ] Generate HTML with proper MathJax config
- [ ] Apply scientific styling
- [ ] Test in browser (check equation rendering)

## Common Pitfalls

1. **Equations not rendering:** Ensure MathJax config includes `'$'` as inline delimiter
2. **Wrong ordering:** Remember sorting must be by tuple (lexicographic), not individual values
3. **Incomplete replacements:** Some equations have multiple atan2 occurrences - must loop until no matches
4. **Nested parentheses:** When extracting atan2/circ_dist arguments, handle nested parens carefully
5. **HTML entity encoding:** Don't HTML-encode equation content (MathJax needs raw LaTeX)

## Validation

After generating HTML, verify:
- [ ] Open in browser and check all equations render properly
- [ ] Row numbers are sequential (1, 2, 3, ...)
- [ ] No `atan2(` or `circ_dist(` remain in equations
- [ ] No `||` (double bars) in equations
- [ ] All free parameters are rendered as LaTeX
- [ ] Table has clean scientific appearance

## File Provided

**`equations_to_html.py`** - Complete, production-ready implementation:
- `EquationsToHTML` class handles all steps
- Optional loss extraction
- Automatic formatting fixes
- Professional HTML generation
- Easy to use and extend

Usage: `python equations_to_html.py`
