import ast
import os
from collections import defaultdict
from pathlib import Path

PACKAGE = Path('ken_burns_reel')
KB_DIR = Path('docs/KB')


def module_tree():
    tree = {}
    for path in PACKAGE.rglob('*.py'):
        rel = path.relative_to(PACKAGE).with_suffix('')
        parts = rel.parts
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node.setdefault('__files__', []).append(rel)
    return tree


def write_architecture(tree, prefix=''):
    lines = []
    for name, subtree in sorted(tree.items()):
        if name == '__files__':
            for f in subtree:
                mod_path = PACKAGE / (str(f) + '.py')
                doc = ast.get_docstring(ast.parse(mod_path.read_text())) or ''
                first = doc.splitlines()[0] if doc else ''
                lines.append(f"- `{f}`: {first}")
        else:
            lines.append(f"{prefix}- {name}/")
            lines.extend(write_architecture(subtree, prefix + '  '))
    return lines


def api_map():
    lines = []
    imports = defaultdict(set)
    for path in PACKAGE.rglob('*.py'):
        mod = path.relative_to(PACKAGE).with_suffix('')
        tree = ast.parse(path.read_text())
        publics = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
                doc = ast.get_docstring(node) or ''
                first = doc.splitlines()[0] if doc else ''
                publics.append((node.name, first))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('ken_burns_reel'):
                    imports[str(mod)].add(node.module)
        if publics:
            lines.append(f"### `{mod}`")
            for name, doc in publics:
                lines.append(f"- `{name}`: {doc}")
    # Imports map
    lines.append("\n## Cross-imports")
    for mod, imps in sorted(imports.items()):
        lines.append(f"- `{mod}` -> {', '.join(sorted(imps))}")
    return lines


def main():
    arch_lines = ['# Architecture'] + write_architecture(module_tree())
    (KB_DIR / 'architecture.md').write_text('\n'.join(arch_lines))
    mod_lines = ['# Modules'] + api_map()
    (KB_DIR / 'modules.md').write_text('\n'.join(mod_lines))


if __name__ == '__main__':
    main()
