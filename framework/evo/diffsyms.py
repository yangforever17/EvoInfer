"""Code regions a commit range touched, in board claim form ("path:Class.method", "path:function", "path:Class",
"path:<module>" for module-level statements other than imports, or bare "path" for new / non-Python files).
Used to post standing board claims for accepted siblings that are waiting to be integrated.

  python -m evo.diffsyms <repo> <base> <head>      -> JSON list on stdout
"""
import ast, json, re, subprocess, sys


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def changed_lines(repo, base, head, path):
    """New-side line numbers touched by the diff (a pure deletion maps to the line it happened at)."""
    out = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", git(repo, "diff", "-U0", base, head, "--", path), re.M):
        start, n = int(m.group(1)), int(m.group(2) or 1)
        out.update(range(start, start + max(n, 1)))
    return out


def spans(tree):
    """(first_line, last_line, qualname) for top-level functions, classes, and methods."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.lineno - len(node.decorator_list), node.end_lineno, node.name, None))
        elif isinstance(node, ast.ClassDef):
            out.append((node.lineno - len(node.decorator_list), node.end_lineno, node.name, None))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((sub.lineno - len(sub.decorator_list), sub.end_lineno, f"{node.name}.{sub.name}", node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append((node.lineno, node.end_lineno, None, "import"))
    return out


def symbols(repo, base, head):
    added = set(git(repo, "diff", "--name-only", "--diff-filter=A", base, head).split())
    where = set()
    for path in git(repo, "diff", "--name-only", "--diff-filter=AMR", base, head).split():
        if path in added or not path.endswith(".py"):
            where.add(path); continue
        try:
            sp = spans(ast.parse(git(repo, "show", f"{head}:{path}")))
        except (SyntaxError, subprocess.CalledProcessError):
            where.add(path); continue
        for ln in changed_lines(repo, base, head, path):
            hits = [s for s in sp if s[0] <= ln <= s[1]]
            if any(h[3] == "import" for h in hits):
                continue
            methods = [h for h in hits if h[3] is not None]          # innermost: a method inside a class
            named = methods or [h for h in hits if h[2]]
            where.add(f"{path}:{named[0][2]}" if named else f"{path}:<module>")
    for path in git(repo, "diff", "--name-only", "--diff-filter=D", base, head).split():
        where.add(path)
    return sorted(where)


if __name__ == "__main__":
    print(json.dumps(symbols(*sys.argv[1:4])))
