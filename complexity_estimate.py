import csv
import ast
import math
import pandas as pd
import io

class ComplexityAnalyzer(ast.NodeVisitor):
    """
    Analyzes a Python function's source code to calculate complexity metrics.
    """
    def __init__(self, source_code):
        self.source_code = source_code
        self.operators = set()
        self.operands = set()
        self.total_operators = 0
        self.total_operands = 0
        self.flop_counts = {
            'binary_ops': 0, # +, -, *
            'div': 0,
            'pow': 0,
            'trig': 0,      # sin, cos, tan
            'arctan2': 0,
            'exp_log': 0,   # exp, log
            'other_func': 0 # clip, maximum, etc.
        }
        self.flop_weights = {
            'binary_ops': 1,
            'div': 4,
            'pow': 8,
            'trig': 10,
            'arctan2': 20,
            'exp_log': 15,
            'other_func': 2
        }
        self.visit(ast.parse(self.source_code))

    def visit_BinOp(self, node):
        self.operators.add(type(node.op).__name__)
        self.total_operators += 1
        
        op_type = type(node.op)
        if op_type in [ast.Add, ast.Sub, ast.Mult]:
            self.flop_counts['binary_ops'] += 1
        elif op_type == ast.Div:
            self.flop_counts['div'] += 1
        elif op_type == ast.Pow:
            self.flop_counts['pow'] += 1
            
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.operators.add(type(node.op).__name__)
        self.total_operators += 1
        self.flop_counts['binary_ops'] += 1 # Treat as a simple op
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.operators.update(type(op).__name__ for op in node.ops)
        self.total_operators += len(node.ops)
        self.flop_counts['binary_ops'] += len(node.ops)
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = ''
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            
        self.operators.add(func_name)
        self.total_operators += 1

        if func_name in ['sin', 'cos', 'tan']:
            self.flop_counts['trig'] += 1
        elif func_name == 'arctan2':
            self.flop_counts['arctan2'] += 1
        elif func_name in ['exp', 'log', 'log2']:
            self.flop_counts['exp_log'] += 1
        else:
            self.flop_counts['other_func'] += 1 # e.g., clip, maximum, abs

        self.generic_visit(node)
        
    def visit_Name(self, node):
        # Only count variable names as operands
        if isinstance(node.ctx, (ast.Load, ast.Store)):
            self.operands.add(node.id)
            self.total_operands += 1

    def visit_Constant(self, node):
        self.operands.add(node.value)
        self.total_operands += 1
        
    def calculate_halstead(self):
        n1 = len(self.operators)
        n2 = len(self.operands)
        N1 = self.total_operators
        N2 = self.total_operands

        if n1 == 0 or n2 == 0 or N1 == 0 or N2 == 0:
            return {
                "vocabulary": 0, "length": 0, "volume": 0,
                "difficulty": 0, "effort": 0
            }

        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 1 else 0
        difficulty = (n1 / 2) * (N2 / n2)
        effort = volume * difficulty
        
        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2)
        }

    def estimate_flops(self):
        total_flops = sum(self.flop_counts[key] * self.flop_weights[key] for key in self.flop_counts)
        return total_flops

def clean_code(code_string):
    """Removes comments and docstrings from a code string."""
    # Remove docstrings
    try:
        parsed = ast.parse(code_string)
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if ast.get_docstring(node):
                    # This is tricky; we'll just remove the docstring lines for length calculation
                    pass
    except SyntaxError:
        return "", 0 # Cannot parse

    lines = code_string.split('\n')
    # Simple removal of single line comments
    no_comments = [line for line in lines if not line.strip().startswith('#')]
    # Crude docstring removal for length
    in_docstring = False
    clean_lines = []
    for line in no_comments:
        if '"""' in line or "'''" in line:
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            clean_lines.append(line)
            
    clean_code_str = "\n".join(clean_lines)
    return clean_code_str, len(clean_code_str.replace(" ", "").replace("\n", ""))

def count_free_parameters(code_string):
    """Counts function arguments with default values."""
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'neuron_model':
                # The first argument 'theta' is not a free parameter
                return len(node.args.defaults)
    except SyntaxError:
        return 0
    return 0


# --- Main execution ---
# Load the CSV file into a pandas DataFrame
# The user should ensure their file is named 'aprograms_db.csv'
try:
    df = pd.read_csv('program_databases/07-23/03-10-12 (text)/combined/programs_db.csv')
    # df = pd.read_csv('program_databases/07-23/02-14-04 (image)/combined/programs_db.csv')
except FileNotFoundError:
    print("Error: 'programs_db.csv' not found. Please save your data to this file.")
    exit()

# Sort by test_loss and get the top programs
df_sorted = df.sort_values(by='test_loss').reset_index()

results = []
for index, row in df_sorted.iterrows():
    program_info = f"Program {index + 1} (Iter: {row['iteration_number']}, Island: {row['birth_island']})"
    code = row['program_code_string']

    # --- Calculations ---
    num_params = count_free_parameters(code)
    clean_code_str, code_len_no_comments = clean_code(code)

    if not clean_code_str:
        # If code is invalid, skip analysis
        results.append({
            "Program": program_info, "Params": num_params, "Code Length": 0,
            "Est. FLOPS": "N/A", "Halstead Volume": "N/A",
            "Halstead Effort": "N/A"
        })
        continue

    analyzer = ComplexityAnalyzer(clean_code_str)
    halstead_metrics = analyzer.calculate_halstead()
    flops = analyzer.estimate_flops()

    results.append({
        "Program": program_info,
        "Params": num_params,
        "Code Length": code_len_no_comments,
        "Est. FLOPS": flops,
        "Halstead Volume": halstead_metrics['volume'],
        "Halstead Effort": halstead_metrics['effort']
    })

# --- Display Results ---
results_df = pd.DataFrame(results)
print("Complexity Analysis of Neuron Models")
print(results_df.to_markdown(index=False))