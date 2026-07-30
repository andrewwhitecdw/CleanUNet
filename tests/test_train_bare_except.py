import ast
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parent.parent / 'train.py'


def test_no_bare_except_in_train_py():
    tree = ast.parse(TRAIN_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    raise AssertionError(
                        "Found bare except clause in train.py"
                    )
