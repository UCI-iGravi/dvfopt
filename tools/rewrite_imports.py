"""One-shot dotted-module-path rewriter for the 0.5.0 reorg.

Usage: python tools/rewrite_imports.py OLD=NEW [OLD=NEW ...] [--dirs d1 d2 ...]
Replaces word-bounded dotted paths in *.py under the given dirs (default: the
sweep set). Longest OLD first so prefixes never clobber (solver3d before solver).
"""

import re
import sys
from pathlib import Path

DEFAULT_DIRS = ["dvfopt", "dvfopt_gui", "tests", "benchmarks", "scripts", "asv_bench", "tools"]


def main() -> None:
    args = sys.argv[1:]
    dirs = DEFAULT_DIRS
    if "--dirs" in args:
        i = args.index("--dirs")
        dirs, args = args[i + 1 :], args[:i]
    pairs = sorted(
        (a.split("=", 1) for a in args), key=lambda p: len(p[0]), reverse=True
    )
    pats = [(re.compile(rf"(?<![\w.]){re.escape(o)}(?![\w])"), n) for o, n in pairs]
    changed = 0
    for d in dirs:
        for f in Path(d).rglob("*.py"):
            text = orig = f.read_text(encoding="utf-8")
            for pat, new in pats:
                text = pat.sub(new, text)
            if text != orig:
                f.write_text(text, encoding="utf-8")
                changed += 1
                print(f"rewrote {f}")
    print(f"{changed} files changed")


if __name__ == "__main__":
    main()
