"""
scan_codebase.py – Argus Codebase Symbol Scanner
==================================================
Scans all Python files under a target directory (defaults to ./argus),
extracts every class, function, method, and async function together with
their full signatures, decorators, and one-liner docstrings, then writes
a structured Markdown report to the directory where the script is run from.

Usage
-----
    python scan_codebase.py [--root <path>] [--out <filename>]

Defaults
--------
    --root  ./argus
    --out   codebase_symbols.md
"""

import ast
import os
import sys
import argparse
import textwrap
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _unparse(node: ast.AST) -> str:
    """Return a source-level string for an AST expression node."""
    try:
        # Python 3.8+
        return ast.unparse(node)
    except AttributeError:
        # Fallback for older Pythons
        return "<expr>"


def _annotation_str(annotation) -> str:
    if annotation is None:
        return ""
    return _unparse(annotation)


def _default_str(default) -> str:
    if default is None:
        return ""
    return _unparse(default)


def _build_signature(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a human-readable signature from an AST function node."""
    args = func_node.args

    # Positional-only  /  regular  *args  **kwargs  keyword-only
    all_args: list[str] = []

    # posonlyargs (Python 3.8+)
    posonlyargs = getattr(args, "posonlyargs", [])
    num_posonly = len(posonlyargs)

    # defaults are right-aligned over (posonlyargs + args)
    regular_args = posonlyargs + args.args
    num_defaults = len(args.defaults)
    defaults_offset = len(regular_args) - num_defaults

    for i, arg in enumerate(regular_args):
        piece = arg.arg
        if ann := _annotation_str(arg.annotation):
            piece += f": {ann}"
        default_idx = i - defaults_offset
        if default_idx >= 0:
            piece += f" = {_default_str(args.defaults[default_idx])}"
        all_args.append(piece)
        if i == num_posonly - 1 and num_posonly:
            all_args.append("/")   # positional-only separator

    # *args
    if args.vararg:
        piece = f"*{args.vararg.arg}"
        if ann := _annotation_str(args.vararg.annotation):
            piece += f": {ann}"
        all_args.append(piece)
    elif args.kwonlyargs:
        all_args.append("*")

    # keyword-only args
    kw_defaults = args.kw_defaults
    for i, arg in enumerate(args.kwonlyargs):
        piece = arg.arg
        if ann := _annotation_str(arg.annotation):
            piece += f": {ann}"
        kd = kw_defaults[i] if i < len(kw_defaults) else None
        if kd is not None:
            piece += f" = {_default_str(kd)}"
        all_args.append(piece)

    # **kwargs
    if args.kwarg:
        piece = f"**{args.kwarg.arg}"
        if ann := _annotation_str(args.kwarg.annotation):
            piece += f": {ann}"
        all_args.append(piece)

    sig = f"({', '.join(all_args)})"
    if ret := _annotation_str(func_node.returns):
        sig += f" -> {ret}"
    return sig


def _get_docstring(node: ast.AST) -> str:
    """Return the first line of the docstring, or empty string."""
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    first_line = doc.strip().splitlines()[0]
    # Cap length for readability in the table
    return first_line[:120] + ("…" if len(first_line) > 120 else "")


def _decorator_names(func_node) -> list[str]:
    names = []
    for d in func_node.decorator_list:
        names.append(_unparse(d))
    return names


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

class SymbolInfo:
    __slots__ = (
        "kind", "name", "qualname", "signature",
        "decorators", "docstring", "lineno", "is_async",
    )

    def __init__(self, kind, name, qualname, signature,
                 decorators, docstring, lineno, is_async=False):
        self.kind = kind           # "class" | "function" | "method"
        self.name = name
        self.qualname = qualname   # Class.method or bare name
        self.signature = signature
        self.decorators = decorators
        self.docstring = docstring
        self.lineno = lineno
        self.is_async = is_async


def extract_symbols(source: str, filepath: str) -> list[SymbolInfo]:
    """Parse *source* and return every class/function/method found."""
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        print(f"  [WARN] SyntaxError in {filepath}: {exc}", file=sys.stderr)
        return []

    symbols: list[SymbolInfo] = []

    def visit_function(node, parent_class: str | None):
        is_async = isinstance(node, ast.AsyncFunctionDef)
        kind = "method" if parent_class else "function"
        qualname = f"{parent_class}.{node.name}" if parent_class else node.name
        sig = _build_signature(node)
        decs = _decorator_names(node)
        doc = _get_docstring(node)
        symbols.append(SymbolInfo(
            kind=kind,
            name=node.name,
            qualname=qualname,
            signature=sig,
            decorators=decs,
            docstring=doc,
            lineno=node.lineno,
            is_async=is_async,
        ))
        # Nested functions inside methods
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_function(child, parent_class=qualname)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = [_unparse(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            doc = _get_docstring(node)
            symbols.append(SymbolInfo(
                kind="class",
                name=node.name,
                qualname=node.name,
                signature=base_str,
                decorators=_decorator_names(node),
                docstring=doc,
                lineno=node.lineno,
            ))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_function(child, parent_class=node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(node, parent_class=None)

    return symbols


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def scan_directory(root: Path) -> dict[str, list[SymbolInfo]]:
    """Walk *root* recursively, return {relative_path: [SymbolInfo, ...]}."""
    results: dict[str, list[SymbolInfo]] = {}
    exclude_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".eggs"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune unwanted directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            full = Path(dirpath) / fname
            rel = full.relative_to(root.parent)
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"  [WARN] Cannot read {full}: {exc}", file=sys.stderr)
                continue
            symbols = extract_symbols(source, str(full))
            if symbols:
                results[str(rel)] = symbols

    return results


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

KIND_ICONS = {
    "class":    "🏛️",
    "function": "🔧",
    "method":   "⚙️",
}

ASYNC_BADGE = "⚡ async "


def _md_escape(text: str) -> str:
    """Minimal Markdown escaping for table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


def generate_report(
    scan_results: dict[str, list[SymbolInfo]],
    root: Path,
    out_path: Path,
) -> None:
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        f"# Argus Codebase Symbol Map",
        f"",
        f"> **Root:** `{root}`  ",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Files scanned:** {len(scan_results)}  ",
        f"> **Total symbols:** "
        f"{sum(len(v) for v in scan_results.values())}",
        f"",
    ]

    # ── Summary statistics ───────────────────────────────────────────────────
    kind_counts: dict[str, int] = defaultdict(int)
    for symbols in scan_results.values():
        for s in symbols:
            kind_counts[s.kind] += 1

    lines += [
        "## Summary",
        "",
        "| Kind | Count |",
        "|------|------:|",
    ]
    for kind in ("class", "function", "method"):
        lines.append(f"| {KIND_ICONS[kind]} {kind.capitalize()} | {kind_counts[kind]} |")
    lines += ["", "---", ""]

    # ── Table of Contents ────────────────────────────────────────────────────
    lines += ["## Table of Contents", ""]
    for rel_path in sorted(scan_results):
        anchor = rel_path.replace("\\", "/").replace("/", "").replace(".", "").replace("_", "").lower()
        display = rel_path.replace("\\", "/")
        lines.append(f"- [{display}](#{anchor})")
    lines += ["", "---", ""]

    # ── Per-file sections ────────────────────────────────────────────────────
    for rel_path in sorted(scan_results):
        symbols = scan_results[rel_path]
        display = rel_path.replace("\\", "/")
        lines += [
            f"## `{display}`",
            "",
        ]

        # Class sections first, then module-level functions
        classes_seen: set[str] = set()

        # Gather class names in file order
        class_symbols = [s for s in symbols if s.kind == "class"]
        func_symbols  = [s for s in symbols if s.kind == "function"]

        for cls in class_symbols:
            classes_seen.add(cls.name)
            decs_str = " ".join(f"`@{d}`" for d in cls.decorators)
            bases_str = cls.signature or "()"
            lines += [
                f"### 🏛️ `class {cls.name}{bases_str}`  <sub>line {cls.lineno}</sub>",
                "",
            ]
            if decs_str:
                lines.append(f"> Decorators: {decs_str}  ")
            if cls.docstring:
                lines.append(f"> {_md_escape(cls.docstring)}  ")
            lines.append("")

            # Methods belonging to this class
            methods = [s for s in symbols
                       if s.kind == "method"
                       and s.qualname.startswith(cls.name + ".")]
            if methods:
                lines += [
                    "| Kind | Name | Signature | Line | Docstring |",
                    "|------|------|-----------|-----:|-----------|",
                ]
                for m in methods:
                    prefix = ASYNC_BADGE if m.is_async else ""
                    decs = ", ".join(f"`@{d}`" for d in m.decorators)
                    dec_col = f" {decs}" if decs else ""
                    doc = _md_escape(m.docstring)
                    sig = _md_escape(m.signature)
                    lines.append(
                        f"| {KIND_ICONS['method']}{dec_col} "
                        f"| `{prefix}{m.name}` "
                        f"| `{sig}` "
                        f"| {m.lineno} "
                        f"| {doc} |"
                    )
                lines.append("")

        # Module-level functions
        if func_symbols:
            lines += [
                "### 🔧 Module-level Functions",
                "",
                "| Kind | Name | Signature | Line | Docstring |",
                "|------|------|-----------|-----:|-----------|",
            ]
            for f in func_symbols:
                prefix = ASYNC_BADGE if f.is_async else ""
                decs = ", ".join(f"`@{d}`" for d in f.decorators)
                dec_col = f" {decs}" if decs else ""
                doc = _md_escape(f.docstring)
                sig = _md_escape(f.signature)
                lines.append(
                    f"| {KIND_ICONS['function']}{dec_col} "
                    f"| `{prefix}{f.name}` "
                    f"| `{sig}` "
                    f"| {f.lineno} "
                    f"| {doc} |"
                )
            lines.append("")

        lines += ["---", ""]

    # ── Flat index ───────────────────────────────────────────────────────────
    lines += [
        "## Flat Symbol Index",
        "",
        "| File | Kind | Qualname | Line |",
        "|------|------|----------|-----:|",
    ]
    for rel_path in sorted(scan_results):
        display = rel_path.replace("\\", "/")
        for s in scan_results[rel_path]:
            icon = KIND_ICONS.get(s.kind, "")
            lines.append(f"| `{display}` | {icon} {s.kind} | `{s.qualname}` | {s.lineno} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅  Report written to: {out_path.resolve()}")
    print(f"    Files   : {len(scan_results)}")
    print(f"    Classes : {kind_counts['class']}")
    print(f"    Functions: {kind_counts['function']}")
    print(f"    Methods : {kind_counts['method']}")
    print(f"    Total   : {sum(kind_counts.values())}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan a Python codebase and produce a symbol-map Markdown report.",
    )
    parser.add_argument(
        "--root",
        default="argus",
        help="Root directory to scan (default: ./argus)",
    )
    parser.add_argument(
        "--out",
        default="codebase_symbols.md",
        help="Output filename – saved in the CWD (default: codebase_symbols.md)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"❌  Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"❌  Path is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path.cwd() / args.out

    print(f"🔍  Scanning: {root}")
    scan_results = scan_directory(root)
    print(f"    Found {len(scan_results)} Python file(s) with symbols.")

    generate_report(scan_results, root, out_path)


if __name__ == "__main__":
    main()
