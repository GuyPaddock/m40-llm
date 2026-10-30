"""Task-template isolation includes code renaming and normalized token spans.

These comparisons add conservative witnesses to the earlier duplicate graph.
They do not equate an automated similarity score with a manual paraphrase pass.
"""

import ast
import hashlib
from collections import defaultdict


def normalized_solution(code: str) -> str | None:
    """Alpha-normalize a single simple function while retaining external names.

    Operators, constants, attribute names and called external functions remain
    significant. Nested scopes/comprehensions are unresolved rather than
    incorrectly renamed as one scope. Reference code is never executed.

    Args:
        code: Inert Python reference solution.

    Returns:
        Normalized digest, or None when scope handling needs manual review.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    unsupported = (ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.ExceptHandler)
    if len(functions) != 1 or any(isinstance(node, unsupported) for node in ast.walk(tree)):
        return None
    function = functions[0]
    if function not in tree.body or function.decorator_list:
        return None
    names: dict[str, str] = {}
    for node in ast.walk(function):
        name = node.arg if isinstance(node, ast.arg) else node.id if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) else None
        if name is not None and name not in names:
            names[name] = f"local_{len(names)}"
    if function.name in names:
        return None
    original_name = function.name
    function.name = "canonical_function"
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            node.arg = names[node.arg]
        elif isinstance(node, ast.Name):
            node.id = names.get(node.id, "canonical_function" if node.id == original_name else node.id)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def shared_span_edges(sequences: dict[int, list[int]]) -> list[list[int]]:
    """Find every pair sharing a normalized contiguous 32-token span.

    Args:
        sequences: Task IDs mapped to runtime-tokenized comparison text.

    Returns:
        Sorted unique source-ID pairs with a shared span.
    """
    owners: dict[tuple[int, ...], set[int]] = defaultdict(set)
    pairs: set[tuple[int, int]] = set()
    for task_id, tokens in sorted(sequences.items()):
        for offset in range(len(tokens) - 31):
            span = tuple(tokens[offset:offset + 32])
            for other in owners[span]:
                pairs.add((other, task_id))
            owners[span].add(task_id)
    return [list(pair) for pair in sorted(pairs)]
