#!/usr/bin/env python3
"""
Spark Performance Pattern Scanner

Scans Python files for patterns that cause performance regressions in Spark 3.3.1,
particularly around DataFrame caching and repeated actions.
"""

import ast
import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set


class SparkPatternDetector(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.cached_vars = {}  # var_name -> (line_number, cache_type)
        self.var_actions = defaultdict(list)  # var_name -> [(action, line_number, code)]
        self.current_function = None
        self.in_loop = False
        self.loop_depth = 0

    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_For(self, node):
        old_loop = self.in_loop
        old_depth = self.loop_depth
        self.in_loop = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.in_loop = old_loop
        self.loop_depth = old_depth

    def visit_While(self, node):
        old_loop = self.in_loop
        old_depth = self.loop_depth
        self.in_loop = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.in_loop = old_loop
        self.loop_depth = old_depth

    def visit_Assign(self, node):
        # Track variable assignments that involve caching
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

            # Check if assignment involves cache() or persist()
            if isinstance(node.value, ast.Call):
                if self._is_cache_call(node.value):
                    cache_type = self._get_cache_type(node.value)
                    self.cached_vars[var_name] = (node.lineno, cache_type)
                else:
                    # Check for chained calls like spark.table().cache() or df.filter().cache()
                    if self._has_cache_in_chain(node.value):
                        cache_type = self._get_cache_type_from_chain(node.value)
                        self.cached_vars[var_name] = (node.lineno, cache_type)
            elif isinstance(node.value, ast.Attribute):
                # Check for chained calls like df.filter().cache()
                if self._has_cache_in_chain(node.value):
                    cache_type = self._get_cache_type_from_chain(node.value)
                    self.cached_vars[var_name] = (node.lineno, cache_type)

        self.generic_visit(node)

    def visit_Call(self, node):
        # Track actions on variables
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                method_name = node.func.attr

                # Check for Spark actions
                if method_name in ['count', 'collect', 'take', 'first', 'show', 'head']:
                    code_line = self._get_source_line(node.lineno)
                    self.var_actions[var_name].append((method_name, node.lineno, code_line, self.in_loop))

                # Check for aggregations and groupBy
                elif method_name in ['agg', 'groupBy', 'groupby', 'sum', 'avg', 'mean', 'max', 'min']:
                    code_line = self._get_source_line(node.lineno)
                    self.var_actions[var_name].append((f'aggregation_{method_name}', node.lineno, code_line, self.in_loop))

                # Check for write operations that trigger computation
                elif method_name in ['write', 'save', 'saveAsTable']:
                    code_line = self._get_source_line(node.lineno)
                    self.var_actions[var_name].append((f'write_{method_name}', node.lineno, code_line, self.in_loop))

            # Check for chained operations like df.filter().count()
            elif self._is_chained_filter_count(node):
                base_var = self._get_base_variable(node.func.value)
                if base_var:
                    code_line = self._get_source_line(node.lineno)
                    self.var_actions[base_var].append(('filter_count', node.lineno, code_line, self.in_loop))

            # Check for any other chained operations that end with actions
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ['count', 'collect', 'take', 'first', 'show', 'head']:
                base_var = self._get_base_variable(node.func.value)
                if base_var and base_var in self.cached_vars:  # Only track if base variable is cached
                    code_line = self._get_source_line(node.lineno)
                    self.var_actions[base_var].append((f'chained_{node.func.attr}', node.lineno, code_line, self.in_loop))

            # Check for chained write operations like df.coalesce().write.parquet()
            elif self._is_chained_write_operation(node):
                base_var = self._get_base_variable_from_chain(node)
                if base_var:
                    code_line = self._get_source_line(node.lineno)
                    write_method = self._get_write_method_from_chain(node)
                    self.var_actions[base_var].append((f'write_{write_method}', node.lineno, code_line, self.in_loop))

        self.generic_visit(node)

    def _is_cache_call(self, node):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ['cache', 'persist']
        return False

    def _get_cache_type(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'cache':
                return 'cache'
            elif node.func.attr == 'persist':
                if node.args:
                    # Check for StorageLevel.MEMORY_AND_DISK
                    arg = node.args[0]
                    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                        if arg.value.id == 'StorageLevel' and arg.attr == 'MEMORY_AND_DISK':
                            return 'persist_memory_and_disk'
                return 'persist'
        return 'unknown'

    def _has_cache_in_chain(self, node):
        current = node
        while True:
            if isinstance(current, ast.Attribute):
                if current.attr in ['cache', 'persist']:
                    return True
                current = current.value
            elif isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute) and current.func.attr in ['cache', 'persist']:
                    return True
                current = current.func
            else:
                break
        return False

    def _get_cache_type_from_chain(self, node):
        current = node
        while True:
            if isinstance(current, ast.Attribute):
                if current.attr == 'cache':
                    return 'cache'
                elif current.attr == 'persist':
                    return 'persist'
                current = current.value
            elif isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute):
                    if current.func.attr == 'cache':
                        return 'cache'
                    elif current.func.attr == 'persist':
                        return 'persist'
                current = current.func
            else:
                break
        return 'unknown'

    def _is_chained_filter_count(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'count':
            if isinstance(node.func.value, ast.Call):
                if isinstance(node.func.value.func, ast.Attribute):
                    return node.func.value.func.attr == 'filter'
        return False

    def _get_base_variable(self, node):
        current = node
        # Handle both Call nodes and Attribute nodes
        while True:
            if isinstance(current, ast.Call):
                current = current.func
            elif isinstance(current, ast.Attribute):
                current = current.value
            elif isinstance(current, ast.Name):
                return current.id
            else:
                break
        return None

    def _is_chained_write_operation(self, node):
        # Check for patterns like df.write.parquet(), df.coalesce().write.csv(), etc.
        if isinstance(node.func, ast.Attribute):
            # Direct write methods like parquet, csv, json, etc.
            if node.func.attr in ['parquet', 'csv', 'json', 'orc', 'text', 'saveAsTable']:
                # Check if this is called on a .write attribute
                if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'write':
                    return True
        return False

    def _get_base_variable_from_chain(self, node):
        # Traverse back through the chain to find the original variable
        current = node
        while hasattr(current, 'func') and hasattr(current.func, 'value'):
            current = current.func.value

        # Continue traversing if we hit an attribute
        while hasattr(current, 'value'):
            current = current.value

        if isinstance(current, ast.Name):
            return current.id
        return None

    def _get_write_method_from_chain(self, node):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return 'unknown'

    def _get_source_line(self, line_number):
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
                if line_number <= len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        return ""

    def analyze_patterns(self):
        # Pattern 1 & 2: Multiple actions on cached DataFrames
        for var_name, (cache_line, cache_type) in self.cached_vars.items():
            actions = self.var_actions.get(var_name, [])

            # Check if this should be reported (multiple actions OR actions in loops)
            loop_actions = [a for a in actions if a[3]]  # in_loop is True
            should_report = len(actions) > 1 or len(loop_actions) > 0

            if should_report:
                severity = min(len(actions), 5)  # Cap at 5

                # Count different types of issues
                count_actions = [a for a in actions if a[0] == 'count']
                filter_count_actions = [a for a in actions if a[0] == 'filter_count']
                agg_actions = [a for a in actions if a[0].startswith('aggregation_')]
                write_actions = [a for a in actions if a[0].startswith('write_')]

                issue_type = "Multiple actions on cached DataFrame"
                details = []

                if len(count_actions) > 1:
                    issue_type = "Multiple count() calls on cached DataFrame"
                    details.append(f"Found {len(count_actions)} count() calls")

                if loop_actions:
                    issue_type = "Actions on cached DataFrame inside loops"
                    details.append(f"Found {len(loop_actions)} actions in loops")

                if filter_count_actions:
                    details.append(f"Found {len(filter_count_actions)} filter().count() patterns")

                if agg_actions:
                    details.append(f"Found {len(agg_actions)} aggregation operations")

                if write_actions:
                    details.append(f"Found {len(write_actions)} write operations")

                if cache_type == 'persist_memory_and_disk':
                    details.append("Uses MEMORY_AND_DISK storage level")
                    severity += 1

                # Get context lines
                context_lines = []
                cache_line_code = self._get_source_line(cache_line)
                context_lines.append(f"  {cache_line_code}")

                for action, line_num, code, in_loop in actions[:5]:  # Show first 5 actions
                    prefix = "  [LOOP] " if in_loop else "  "
                    context_lines.append(f"{prefix}{code}")

                if len(actions) > 5:
                    context_lines.append(f"  ... and {len(actions) - 5} more actions")

                self.issues.append({
                    'file': self.filename,
                    'line': cache_line,
                    'type': issue_type,
                    'severity': severity,
                    'details': details,
                    'context': context_lines,
                    'var_name': var_name
                })


def scan_file(filepath: str) -> List[Dict]:
    """Scan a single Python file for Spark performance patterns."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=filepath)
        detector = SparkPatternDetector(filepath)
        detector.visit(tree)
        detector.analyze_patterns()

        return detector.issues
    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)
        return []


def scan_directory(directory: str) -> List[Dict]:
    """Recursively scan directory for Python files."""
    all_issues = []

    for root, dirs, files in os.walk(directory):
        # Skip common irrelevant directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                issues = scan_file(filepath)
                all_issues.extend(issues)

    return all_issues


def format_output(issues: List[Dict]) -> None:
    """Format and print the detected issues."""
    if not issues:
        print("No Spark performance patterns detected.")
        return

    # Sort by severity (descending) then by file
    issues.sort(key=lambda x: (-x['severity'], x['file'], x['line']))

    print(f"Found {len(issues)} potential performance issues:\n")

    for issue in issues:
        severity_stars = "⭐" * issue['severity']
        print(f"{issue['file']}:{issue['line']} - {issue['type']} {severity_stars}")

        for line in issue['context']:
            print(line)

        if issue['details']:
            for detail in issue['details']:
                print(f"  └─ {detail}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Scan Python files for Spark performance regression patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Scan current directory
  %(prog)s /path/to/spark/repo  # Scan specific directory
  %(prog)s --file myfile.py    # Scan single file
        """
    )

    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory to scan (default: current directory)'
    )

    parser.add_argument(
        '--file',
        help='Scan a single Python file instead of directory'
    )

    parser.add_argument(
        '--min-severity',
        type=int,
        default=1,
        help='Minimum severity to report (1-5, default: 1)'
    )

    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        issues = scan_file(args.file)
    else:
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}", file=sys.stderr)
            sys.exit(1)
        issues = scan_directory(args.directory)

    # Filter by minimum severity
    filtered_issues = [issue for issue in issues if issue['severity'] >= args.min_severity]

    format_output(filtered_issues)

    if filtered_issues:
        print(f"\nSummary: {len(filtered_issues)} issues found")
        severity_counts = defaultdict(int)
        for issue in filtered_issues:
            severity_counts[issue['severity']] += 1

        for severity in sorted(severity_counts.keys(), reverse=True):
            stars = "⭐" * severity
            print(f"  Severity {severity} {stars}: {severity_counts[severity]} issues")


if __name__ == "__main__":
    main()
