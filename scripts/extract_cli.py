import ast
from pathlib import Path

MAIN = Path('ken_burns_reel/__main__.py')
KB = Path('docs/KB/cli_reference.md')


def extract():
    tree = ast.parse(MAIN.read_text())
    args = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            flag = node.args[0].value if node.args else ''
            default = None
            help_str = ''
            arg_type = 'str'
            for kw in node.keywords:
                if kw.arg == 'default':
                    try:
                        default = ast.literal_eval(kw.value)
                    except Exception:  # noqa: BLE001
                        default = ast.unparse(kw.value)
                elif kw.arg == 'help':
                    help_str = kw.value.value if isinstance(kw.value, ast.Constant) else ''
                elif kw.arg == 'type':
                    arg_type = getattr(kw.value, 'id', 'str')
                elif kw.arg == 'choices':
                    default = f"choices={ast.literal_eval(kw.value)}"
            args.append((flag, arg_type, default, help_str))
    lines = ['# CLI Reference', '| Flag | Type | Default | Help |', '|------|------|---------|------|']
    for flag, arg_type, default, help_str in args:
        lines.append(f"| {flag} | {arg_type} | {default} | {help_str} |")
    KB.write_text('\n'.join(lines))


if __name__ == '__main__':
    extract()
