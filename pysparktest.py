#!/usr/bin/env python3
"""
Spark Performance Pattern Scanner

Finds count() calls and backtracks to identify performance regression patterns.
Focuses on finding cached DataFrames with multiple actions across files.
"""

import ast
import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set


class DataFrameTracker:
    def __init__(self):
        # Track all performance-degrading operations
        self.performance_operations = []  # (file, line, var_name, operation_type, full_context)

        # Specific operation types
        self.action_calls = []       # count, collect, show, take, first, head, foreach
        self.write_calls = []        # write, save, saveAsTable + format methods
        self.cache_calls = []        # cache, persist, checkpoint
        self.expensive_ops = []      # join, union, groupBy with actions

        # Track variable assignments across files
        self.variable_assignments = {}  # var_name -> [(file, line, assignment_context)]

        # Track method calls on variables
        self.method_calls = defaultdict(list)  # var_name -> [(file, line, method, context)]

        # Track function calls that might return DataFrames
        self.function_calls = []  # (file, line, func_name, assigned_to, context)

        # All cached variables
        self.cached_variables = {}  # var_name -> (file, line, cache_type, context)


class SparkPatternDetector(ast.NodeVisitor):
    def __init__(self, filename: str, tracker: DataFrameTracker):
        self.filename = filename
        self.tracker = tracker
        self.current_function = None

    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            context = self._get_context(node.lineno)

            # Track all assignments
            if var_name not in self.tracker.variable_assignments:
                self.tracker.variable_assignments[var_name] = []
            self.tracker.variable_assignments[var_name].append((self.filename, node.lineno, context))

            # Check for caching patterns
            if self._involves_caching(node.value):
                cache_type = self._get_cache_type(node.value)
                self.tracker.cached_variables[var_name] = (self.filename, node.lineno, cache_type, context)

            # Track function calls that might return DataFrames
            if isinstance(node.value, ast.Call):
                func_name = self._get_function_name(node.value)
                if func_name:
                    self.tracker.function_calls.append((self.filename, node.lineno, func_name, var_name, context))

        self.generic_visit(node)

    def visit_Call(self, node):
        # Track all performance-degrading operations
        var_name = self._get_variable_from_call(node)
        context = self._get_context(node.lineno, lines_before=3, lines_after=1)

        # Detect different types of performance operations
        operation_type = self._classify_operation(node)
        if operation_type:
            self.tracker.performance_operations.append((self.filename, node.lineno, var_name, operation_type, context))

            # Store in specific lists for detailed analysis
            if operation_type in ['count', 'collect', 'show', 'take', 'first', 'head', 'foreach']:
                self.tracker.action_calls.append((self.filename, node.lineno, var_name, operation_type, context))
            elif operation_type.startswith('write_') or operation_type in ['write', 'save', 'saveAsTable']:
                self.tracker.write_calls.append((self.filename, node.lineno, var_name, operation_type, context))
            elif operation_type in ['cache', 'persist', 'checkpoint']:
                self.tracker.cache_calls.append((self.filename, node.lineno, var_name, operation_type, context))
            elif operation_type.startswith('expensive_'):
                self.tracker.expensive_ops.append((self.filename, node.lineno, var_name, operation_type, context))

        # Track all method calls on variables
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var_name = node.func.value.id
            method_name = node.func.attr
            context = self._get_context(node.lineno)
            self.tracker.method_calls[var_name].append((self.filename, node.lineno, method_name, context))

        self.generic_visit(node)

    def _classify_operation(self, node):
        """Classify the type of performance operation"""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr

        # Action operations (trigger computation)
        if method_name in ['count', 'collect', 'show', 'take', 'first', 'head', 'foreach', 'reduce']:
            return method_name

        # Write operations
        if method_name in ['write', 'save', 'saveAsTable']:
            return method_name

        # Write format operations (parquet, csv, etc.)
        if method_name in ['parquet', 'csv', 'json', 'orc', 'text'] and self._is_write_format_call(node):
            return f'write_{method_name}'

        # Cache operations
        if method_name in ['cache', 'persist', 'checkpoint']:
            return method_name

        # Expensive transformations that often lead to performance issues
        if method_name in ['join', 'union', 'unionAll']:
            return f'expensive_{method_name}'

        # GroupBy operations (especially when followed by actions)
        if method_name in ['groupBy', 'groupby', 'agg']:
            return f'expensive_{method_name}'

        return None

    def _is_write_format_call(self, node):
        """Check if this format method is called on a write object"""
        if isinstance(node.func.value, ast.Attribute):
            return node.func.value.attr == 'write'
        return False

    def _get_variable_from_call(self, node):
        """Extract the base variable name from a method call (including chained calls)"""
        current = node.func
        while isinstance(current, ast.Attribute):
            current = current.value

        # Handle chained calls like df.filter().count()
        while isinstance(current, ast.Call):
            current = current.func
            while isinstance(current, ast.Attribute):
                current = current.value

        if isinstance(current, ast.Name):
            return current.id
        return None

    def _involves_caching(self, node):
        """Check if assignment involves any caching method"""
        cache_methods = ['cache', 'persist', 'checkpoint']
        return self._has_method_in_chain(node, cache_methods)

    def _get_cache_type(self, node):
        """Get the type of caching used"""
        cache_methods = ['cache', 'persist', 'checkpoint']
        method = self._find_method_in_chain(node, cache_methods)

        if method == 'persist':
            # Check for StorageLevel
            if self._has_storage_level(node):
                return 'persist_with_storage_level'
            return 'persist'
        return method or 'unknown'

    def _has_method_in_chain(self, node, methods):
        """Check if any of the methods appear in the call chain"""
        current = node
        while current:
            if isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute) and current.func.attr in methods:
                    return True
                current = current.func
            elif isinstance(current, ast.Attribute):
                if current.attr in methods:
                    return True
                current = current.value
            else:
                break
        return False

    def _find_method_in_chain(self, node, methods):
        """Find which method from the list appears in the call chain"""
        current = node
        while current:
            if isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute) and current.func.attr in methods:
                    return current.func.attr
                current = current.func
            elif isinstance(current, ast.Attribute):
                if current.attr in methods:
                    return current.attr
                current = current.value
            else:
                break
        return None

    def _has_storage_level(self, node):
        """Check if persist() uses StorageLevel"""
        # This is a simplified check - could be enhanced
        context = self._get_context_for_node(node)
        return 'StorageLevel' in context

    def _get_function_name(self, node):
        """Get function name from a call"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _get_context(self, line_number, lines_before=1, lines_after=1):
        """Get surrounding code context for a line"""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            start = max(0, line_number - lines_before - 1)
            end = min(len(lines), line_number + lines_after)

            context_lines = []
            for i in range(start, end):
                marker = " >>> " if i == line_number - 1 else "     "
                context_lines.append(f"{marker}{lines[i].rstrip()}")

            return '\n'.join(context_lines)
        except:
            return f"Line {line_number}: <could not read context>"

    def _get_context_for_node(self, node):
        """Get code context for a specific node"""
        return self._get_context(node.lineno)


def analyze_patterns(tracker: DataFrameTracker):
    """Analyze collected data to find performance patterns"""
    issues = []

    print(f"🔍 Found {len(tracker.performance_operations)} performance operations:")
    print(f"  • {len(tracker.action_calls)} action calls (count, collect, show, etc.)")
    print(f"  • {len(tracker.write_calls)} write operations")
    print(f"  • {len(tracker.cache_calls)} cache/persist operations")
    print(f"  • {len(tracker.expensive_ops)} expensive operations (join, groupBy, etc.)")
    print(f"📦 Found {len(tracker.cached_variables)} cached variables")
    print(f"📋 Tracking {len(tracker.variable_assignments)} variables across files\n")

    # Analyze each performance operation
    for file, line, var_name, operation_type, context in tracker.performance_operations:
        issue = analyze_performance_operation(tracker, file, line, var_name, operation_type, context)
        if issue:
            issues.append(issue)

    return issues


def analyze_performance_operation(tracker: DataFrameTracker, file: str, line: int, var_name: str, operation_type: str, context: str):
    """Analyze a specific performance operation to find related patterns"""
    if not var_name:
        return None

    issue = {
        'file': file,
        'line': line,
        'type': f'{operation_type} operation analysis',
        'operation_type': operation_type,
        'var_name': var_name,
        'context': context,
        'patterns': [],
        'severity': 1
    }

    # Base severity based on operation type
    if operation_type in ['collect', 'write', 'save']:
        issue['severity'] = 3  # High impact operations
    elif operation_type in ['count', 'show', 'take']:
        issue['severity'] = 2  # Medium impact
    elif operation_type.startswith('expensive_'):
        issue['severity'] = 2  # Expensive transformations

    # Check if this variable is cached
    if var_name in tracker.cached_variables:
        cache_file, cache_line, cache_type, cache_context = tracker.cached_variables[var_name]
        issue['patterns'].append(f"Variable is cached ({cache_type}) at {cache_file}:{cache_line}")
        issue['cache_info'] = (cache_file, cache_line, cache_context)
        issue['severity'] += 2

    # Find other actions on the same variable
    actions = tracker.method_calls.get(var_name, [])
    action_methods = ['count', 'collect', 'show', 'take', 'first', 'head', 'foreach']

    related_actions = [(f, l, m, c) for f, l, m, c in actions if m in action_methods]

    if len(related_actions) > 1:
        issue['patterns'].append(f"Found {len(related_actions)} actions on same variable")
        issue['related_actions'] = related_actions
        issue['severity'] += len(related_actions) - 1

    # Check for actions in proximity (same file, nearby lines)
    nearby_actions = []
    for f, l, m, c in related_actions:
        if f == file and abs(l - line) <= 20 and l != line:  # Within 20 lines
            nearby_actions.append((f, l, m, c))

    if nearby_actions:
        issue['patterns'].append(f"Found {len(nearby_actions)} actions nearby (within 20 lines)")
        issue['nearby_actions'] = nearby_actions
        issue['severity'] += 1

    # Check variable assignment history
    if var_name in tracker.variable_assignments:
        assignments = tracker.variable_assignments[var_name]
        if len(assignments) > 1:
            issue['patterns'].append(f"Variable reassigned {len(assignments)} times across files")
            issue['assignments'] = assignments

    # Only report if we found interesting patterns
    if len(issue['patterns']) > 0:
        return issue

    return None


def analyze_cache_usage(tracker: DataFrameTracker, issue: dict, var_name: str):
    """Special analysis for cache/persist operations"""
    # Find all subsequent operations on this cached variable
    cache_line = issue['line']
    subsequent_ops = []

    for f, l, v, op, ctx in tracker.performance_operations:
        if v == var_name and l > cache_line:
            subsequent_ops.append((f, l, op, ctx))

    if subsequent_ops:
        action_ops = [op for f, l, op, ctx in subsequent_ops if op in ['count', 'collect', 'show', 'take', 'first']]
        write_ops = [op for f, l, op, ctx in subsequent_ops if op.startswith('write_') or op in ['write', 'save']]

        if len(action_ops) > 1:
            issue['patterns'].append(f"Cached variable used in {len(action_ops)} subsequent actions - potential recomputation")
            issue['severity'] += 2

        if write_ops:
            issue['patterns'].append(f"Cached variable written {len(write_ops)} times - may not benefit from caching")
            issue['severity'] += 1


def analyze_write_patterns(tracker: DataFrameTracker, issue: dict, var_name: str):
    """Special analysis for write operations"""
    # Check if this DataFrame was cached before writing
    if var_name in tracker.cached_variables:
        cache_file, cache_line, cache_type, cache_context = tracker.cached_variables[var_name]
        if cache_line < issue['line']:
            issue['patterns'].append(f"Writing cached DataFrame - caching may be unnecessary")
            issue['severity'] += 1

    # Check for multiple writes of the same variable
    write_ops = [(f, l, op, ctx) for f, l, v, op, ctx in tracker.write_calls if v == var_name]
    if len(write_ops) > 1:
        issue['patterns'].append(f"Variable written {len(write_ops)} times")
        issue['severity'] += 1


def analyze_action_patterns(tracker: DataFrameTracker, issue: dict, var_name: str):
    """Special analysis for action operations (count, collect, etc.)"""
    # Check for repeated actions on cached DataFrames
    if var_name in tracker.cached_variables:
        cache_file, cache_line, cache_type, cache_context = tracker.cached_variables[var_name]

        # Count actions after caching
        actions_after_cache = []
        for f, l, v, op, ctx in tracker.action_calls:
            if v == var_name and l > cache_line:
                actions_after_cache.append((f, l, op, ctx))

        if len(actions_after_cache) > 1:
            issue['patterns'].append(f"Multiple actions on cached DataFrame - good caching pattern")
            issue['severity'] -= 1  # This is actually good!
        elif len(actions_after_cache) == 1 and actions_after_cache[0][2] == issue['operation_type']:
            issue['patterns'].append(f"Single action on cached DataFrame - caching may be unnecessary")
            issue['severity'] += 1
    else:
        # Action on non-cached DataFrame - check if it should be cached
        action_ops = [(f, l, op, ctx) for f, l, v, op, ctx in tracker.action_calls if v == var_name]
        if len(action_ops) > 1:
            issue['patterns'].append(f"Multiple actions on non-cached DataFrame - consider caching")
            issue['severity'] += 2


def scan_repository(repo_path: str):
    """Scan entire repository for patterns"""
    tracker = DataFrameTracker()

    # Scan all Python files
    for root, dirs, files in os.walk(repo_path):
        # Skip common irrelevant directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', 'env']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    tree = ast.parse(content, filename=filepath)
                    detector = SparkPatternDetector(filepath, tracker)
                    detector.visit(tree)

                except Exception as e:
                    print(f"Error scanning {filepath}: {e}")

    return analyze_patterns(tracker)


def format_output(issues: List[Dict]):
    """Format and display the results"""
    if not issues:
        print("No performance patterns detected.")
        return

    # Sort by severity
    issues.sort(key=lambda x: x['severity'], reverse=True)

    print(f"🚨 Found {len(issues)} potential performance issues:\n")

    for i, issue in enumerate(issues, 1):
        severity_stars = "⭐" * min(issue['severity'], 5)
        print(f"{i}. {issue['file']}:{issue['line']} - {issue['operation_type']} on '{issue['var_name']}' {severity_stars}")

        for pattern in issue['patterns']:
            print(f"   └─ {pattern}")

        print(f"\n   Context:")
        for line in issue['context'].split('\n'):
            print(f"   {line}")

        # Show cache info if available
        if 'cache_info' in issue:
            cache_file, cache_line, cache_context = issue['cache_info']
            print(f"\n   Cache location ({cache_file}:{cache_line}):")
            for line in cache_context.split('\n'):
                print(f"   {line}")

        # Show related operations
        if 'related_operations' in issue:
            print(f"\n   Related operations:")
            for f, l, op, c in issue['related_operations'][:5]:  # Show first 5
                print(f"   • {f}:{l} - {op}()")

        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan repository for Spark performance patterns by backtracking from count() calls",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'repo_path',
        nargs='?',
        default='.',
        help='Repository path to scan (default: current directory)'
    )

    parser.add_argument(
        '--operations', '-o',
        choices=['all', 'actions', 'writes', 'cache'],
        default='all',
        help='Filter by operation type (default: all)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        print(f"Path not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Scanning repository: {args.repo_path}")
    print("Looking for performance-degrading operations and analyzing patterns...\n")

    issues = scan_repository(args.repo_path)

    # Filter by operation type if specified
    if args.operations != 'all':
        filtered_issues = []
        for issue in issues:
            op_type = issue['operation_type']
            if args.operations == 'actions' and op_type in ['count', 'collect', 'show', 'take', 'first', 'head']:
                filtered_issues.append(issue)
            elif args.operations == 'writes' and (op_type.startswith('write_') or op_type in ['write', 'save', 'saveAsTable']):
                filtered_issues.append(issue)
            elif args.operations == 'cache' and op_type in ['cache', 'persist', 'checkpoint']:
                filtered_issues.append(issue)
        issues = filtered_issues
        print(f"\nFiltered to {len(issues)} {args.operations} operations\n")

    format_output(issues)

    if issues:
        severity_counts = defaultdict(int)
        for issue in issues:
            severity_counts[min(issue['severity'], 5)] += 1

        print(f"Summary: {len(issues)} issues found")
        for severity in sorted(severity_counts.keys(), reverse=True):
            stars = "⭐" * severity
            print(f"  Severity {severity} {stars}: {severity_counts[severity]} issues")


if __name__ == "__main__":
    main()
