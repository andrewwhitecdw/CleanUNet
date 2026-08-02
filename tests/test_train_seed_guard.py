import ast
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parent.parent / 'train.py'


def test_seed_calls_not_at_module_level():
    tree = ast.parse(TRAIN_PY.read_text())
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        name = None
        if isinstance(call.func, ast.Attribute):
            name = call.func.attr
        elif isinstance(call.func, ast.Name):
            name = call.func.id
        assert name not in ('seed', 'manual_seed'), (
            f"Seed call {name} found at module level"
        )
