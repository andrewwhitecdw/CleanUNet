import ast
import os
import unittest


class TestNoBareExcept(unittest.TestCase):
    def test_train_py_has_no_bare_except(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
        with open(path) as f:
            tree = ast.parse(f.read())
        bare_handlers = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        self.assertEqual(bare_handlers, [], "train.py contains bare except: handlers")

