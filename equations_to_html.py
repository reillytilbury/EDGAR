"""
Convert equations.md to a beautiful scientific HTML table.

This script processes an equations.md file (HTML table format with Iter, Island, Batch,
Free Params, Equation columns) and generates a clean, publication-ready HTML table.

Features:
- Sorts equations by (iter, island, batch) tuple
- Simple sequential row numbering (1, 2, 3, ...)
- Removes Appearance Index column
- Fixes complex mathematical expressions (atan2, circ_dist, double bars)
- Configures MathJax for proper LaTeX rendering
- Professional scientific styling (Times New Roman, black/white only)
- Optional loss values from log file
"""

import re
from pathlib import Path
from typing import Optional, Dict, Tuple


class EquationsToHTML:
    """Convert equations.md to HTML with formatting fixes."""

    def __init__(self, equations_md_path: str, output_html_path: str, log_path: Optional[str] = None):
        """
        Initialize the converter.

        Args:
            equations_md_path: Path to equations.md (HTML table format)
            output_html_path: Path to write output HTML file
            log_path: Optional path to hypothesis_engine.log for extracting losses
        """
        self.equations_md_path = equations_md_path
        self.output_html_path = output_html_path
        self.log_path = log_path
        self.losses_map = {}

        if log_path:
            self._extract_losses_from_log()

    def _extract_losses_from_log(self) -> None:
        """Extract loss values from hypothesis_engine.log by (iter, island, batch)."""
        if not self.log_path:
            return

        with open(self.log_path, "r") as f:
            text = f.read()

        for match in re.finditer(
            r"id=(\d+),(\d+),(\d+)\n.*?(?:id=\d+,\d+,\d+\n|$)",
            text,
            re.DOTALL
        ):
            iter_val = int(match.group(1))
            island = int(match.group(2))
            batch = int(match.group(3))
            key = (iter_val, island, batch)

            # Find Loss in this block
            block = match.group(0)
            loss_match = re.search(r"Loss:\s*([\d.]+)", block)
            if loss_match:
                self.losses_map[key] = float(loss_match.group(1))

    def _parse_equations_md(self) -> list:
        """Parse HTML table from equations.md and extract rows."""
        with open(self.equations_md_path, "r") as f:
            content = f.read()

        rows = []
        for match in re.finditer(
            r"<tr><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td><td>(.*?)</td><td>(.*?)</td></tr>",
            content,
            re.DOTALL
        ):
            iter_val, island, batch, free_params, equation = match.groups()
            key = (int(iter_val), int(island), int(batch))
            loss = self.losses_map.get(key)

            rows.append({
                "index": key,
                "free_params": free_params.strip(),
                "equation": equation.strip(),
                "loss": loss
            })

        return rows

    def _sort_rows(self, rows: list) -> list:
        """Sort rows by (iter, island, batch) tuple."""
        return sorted(rows, key=lambda x: x["index"])

    def _fix_atan2_expressions(self, equation: str) -> str:
        """Replace atan2(sin(...), cos(...)) with |...|"""
        def replace_atan2(content):
            count = 0
            while True:
                match = re.search(
                    r'\\operatorname\{atan2\}\(\\sin\(([^()]*(?:\([^()]*\)[^()]*)*)\),\s*\\cos\(\1\)\)',
                    content
                )
                if not match:
                    break
                angle_expr = match.group(1)
                content = content[:match.start()] + f"|{angle_expr}|" + content[match.end():]
                count += 1
            return content, count

        new_equation, _ = replace_atan2(equation)
        return new_equation

    def _fix_circ_dist_expressions(self, equation: str) -> str:
        """Replace circ_dist(..., ...) with |... - ...|"""
        while True:
            match = re.search(
                r'\\operatorname\{circ\\_dist\}\(([^,]+),\s*([^)]+(?:\([^)]*\)[^)]*)*)\)',
                equation
            )
            if not match:
                break
            arg1 = match.group(1).strip()
            arg2 = match.group(2).strip()
            equation = equation[:match.start()] + f"|{arg1} - ({arg2})|" + equation[match.end():]
        return equation

    def _fix_double_bars(self, equation: str) -> str:
        """Replace || with |"""
        return equation.replace("||", "|")

    def _fix_formatting(self, equation: str) -> str:
        """Apply all formatting fixes."""
        equation = self._fix_atan2_expressions(equation)
        equation = self._fix_circ_dist_expressions(equation)
        equation = self._fix_double_bars(equation)
        return equation

    def _count_params(self, param_str: str) -> int:
        """Count number of free parameters from LaTeX string."""
        import re
        matches = re.findall(r'\$[^$]+\$', param_str)
        return len(matches)

    def _classify_differentiability(self, equation: str) -> str:
        """Classify differentiability class of an equation."""
        import re
        # Check for non-smooth operations
        if 'max(' in equation:
            return '1'
        if 'operatorname{sign}' in equation or '\\text{sgn}' in equation:
            return '1'

        if '|' not in equation:
            return '∞'

        # If equation has )^2 with absolute values, likely smooth (|x|^2 = x^2)
        if (r'\right)^2' in equation or r'\right)^{2}' in equation or
                (')^2' in equation and '|' in equation)):
            return '∞'

        # Otherwise |x| without even power is at most C^1
        if '|' in equation:
            return '1'

        return '∞'

    def _generate_html(self, rows: list) -> str:
        """Generate HTML table with MathJax support."""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Discovered Equations and Free Parameters</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            margin: 20px 40px;
            background-color: white;
            color: #333;
            line-height: 1.5;
        }
        h1 {
            color: #000;
            font-size: 1.8em;
            margin-bottom: 20px;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
            table-layout: fixed;
        }
        th, td {
            border-top: 1px solid #000;
            border-bottom: 1px solid #000;
            padding: 8px 6px;
            text-align: left;
            font-size: 0.85em;
            vertical-align: top;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        th {
            background-color: white;
            color: #000;
            font-weight: bold;
            border-bottom: 2px solid #000;
            text-align: center;
        }
        tr:last-child td {
            border-bottom: 2px solid #000;
        }
        .program-id {
            font-weight: normal;
            color: #000;
        }
        .loss {
            text-align: center;
            font-weight: normal;
        }
        .equation {
            font-size: 0.80em;
        }
        th:nth-child(1), td:nth-child(1) { width: 8.3%; text-align: center; }
        th:nth-child(2), td:nth-child(2) { width: 8.3%; }
        th:nth-child(3), td:nth-child(3) { width: 8.3%; text-align: center; }
        th:nth-child(4), td:nth-child(4) { width: 8.3%; }
        th:nth-child(5), td:nth-child(5) { width: 8.3%; }
        th:nth-child(6), td:nth-child(6) { width: 8.3%; text-align: center; }
        th:nth-child(7), td:nth-child(7) { width: 8.3%; text-align: center; }
        th:nth-child(8), td:nth-child(8) { width: 50%; }
    </style>
</head>
<body>
    <h1>Discovered Equations and Free Parameters</h1>
    <table>
        <thead>
            <tr>
                <th>Program</th>
                <th>Free Parameters</th>
                <th>N Free Params</th>
                <th>CV MSE</th>
                <th>Complexity Penalised Loss</th>
                <th>Differentiability Class</th>
                <th>Equation</th>
            </tr>
        </thead>
        <tbody>
"""

        # Add data rows
        for row_num, row in enumerate(rows, start=1):
            num_params = self._count_params(row['free_params'])
            diff_class = self._classify_differentiability(row['equation'])

            loss = row['loss'] if row['loss'] is not None else 0
            cv_mse = loss - 0.01 * num_params

            html += f"""            <tr>
                <td class="program-id">{row_num}</td>
                <td>{row['free_params']}</td>
                <td class="program-id">{num_params}</td>
                <td class="loss">{cv_mse:.2f}</td>
                <td class="loss">{loss:.2f}</td>
                <td class="program-id">{diff_class}</td>
                <td class="equation">{row['equation']}</td>
            </tr>
"""

        html += """        </tbody>
    </table>
</body>
</html>
"""
        return html

    def convert(self) -> None:
        """Execute the full conversion pipeline."""
        print(f"Reading {self.equations_md_path}...")
        rows = self._parse_equations_md()
        print(f"Parsed {len(rows)} equations")

        print("Sorting by (iter, island, batch)...")
        rows = self._sort_rows(rows)

        print("Fixing formatting issues...")
        for row in rows:
            row["equation"] = self._fix_formatting(row["equation"])

        print("Generating HTML...")
        html = self._generate_html(rows)

        print(f"Writing to {self.output_html_path}...")
        with open(self.output_html_path, "w") as f:
            f.write(html)

        print(f"✓ Done! Generated {len(rows)} equations")
        if self.losses_map:
            print(f"✓ Included loss values for {sum(1 for r in rows if r['loss'] is not None)} equations")


def main():
    """Example usage."""
    converter = EquationsToHTML(
        equations_md_path="equations.md",
        output_html_path="discovered_equations.html",
        log_path=None  # Set to log file path if losses are available
    )
    converter.convert()


if __name__ == "__main__":
    main()
