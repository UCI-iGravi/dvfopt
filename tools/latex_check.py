#!/usr/bin/env python3
"""Install-free LaTeX sanity checker (substitutes for a local pdflatex compile).

These manuscripts are authored in Overleaf, so no local TeX toolchain exists.
This checker catches the structural errors a compile would catch:
  - mismatched \\begin{env} / \\end{env}
  - unbalanced braces (outside comments, ignoring escaped \\{ \\})
  - \\ref/\\autoref/\\eqref/\\cref to labels that do not exist
  - \\cite to bib keys that do not exist
  - duplicate \\label definitions

It does NOT validate macro semantics or render output — the author runs the
real compile in Overleaf. Exit code is non-zero if any ERROR is found
(undefined refs/cites, env/brace mismatch, duplicate labels). Warnings
(unused labels) do not fail.

Usage:
  python tools/latex_check.py writing/tvcg/tvcg_manuscript.tex
  # follows one level of \\input{} (e.g. dvfopt) and reads sibling .bib files
"""
import os
import re
import sys
from pathlib import Path

COMMENT_RE = re.compile(r'(?<!\\)%.*')
# verbatim-like environments whose body must NOT be parsed as live LaTeX
VERBATIM_ENVS = ('comment', 'verbatim', 'lstlisting')


def strip_comments(text: str) -> str:
    # 1) drop the body of comment/verbatim environments (keep line count stable)
    for env in VERBATIM_ENVS:
        pat = re.compile(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', re.DOTALL)
        text = pat.sub(lambda m: '\n' * m.group(0).count('\n'), text)
    # 2) drop line comments (unescaped %)
    return '\n'.join(COMMENT_RE.sub('', line) for line in text.splitlines())


def read_tex(root_path: str):
    """Read root + one level of \\input{} files (skipping commented inputs)."""
    base = os.path.dirname(root_path)
    raw = Path(root_path).read_text(encoding='utf-8')
    src = strip_comments(raw)
    chunks = [(root_path, src)]
    for m in re.finditer(r'\\input\{([^}]+)\}', src):
        name = m.group(1).strip()
        if not name.endswith('.tex'):
            name += '.tex'
        p = os.path.join(base, name)
        if os.path.exists(p):
            chunks.append((p, strip_comments(Path(p).read_text(encoding='utf-8'))))
    return chunks


def bib_keys(root_path: str):
    base = os.path.dirname(root_path)
    keys = set()
    for fn in os.listdir(base or '.'):
        if fn.endswith('.bib'):
            txt = Path(os.path.join(base, fn)).read_text(encoding='utf-8', errors='ignore')
            keys |= set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', txt))
    return keys


def check(root_path: str) -> int:
    chunks = read_tex(root_path)
    errors, warnings = [], []

    labels, dup_labels = set(), []
    refs, cites = [], []
    env_stack = []  # (env, file, lineno)
    brace = 0

    for path, src in chunks:
        fname = os.path.basename(path)
        for m in re.finditer(r'\\label\{([^}]+)\}', src):
            k = m.group(1)
            if k in labels:
                dup_labels.append(k)
            labels.add(k)
        for cmd in (r'\\ref', r'\\autoref', r'\\eqref', r'\\cref', r'\\Cref'):
            for m in re.finditer(cmd + r'\{([^}]+)\}', src):
                refs.append(m.group(1))
        for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]+)\}', src):
            for k in m.group(1).split(','):
                k = k.strip()
                if k:
                    cites.append(k)
        # environments + braces, line-tracked for useful messages
        for i, line in enumerate(src.splitlines(), 1):
            for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', line):
                kind, env = m.group(1), m.group(2)
                if kind == 'begin':
                    env_stack.append((env, fname, i))
                else:
                    if not env_stack:
                        errors.append(f'{fname}:{i}: \\end{{{env}}} with no open environment')
                    elif env_stack[-1][0] != env:
                        oe, of, ol = env_stack[-1]
                        errors.append(f'{fname}:{i}: \\end{{{env}}} but innermost open is \\begin{{{oe}}} ({of}:{ol})')
                        env_stack.pop()
                    else:
                        env_stack.pop()
            cleaned = re.sub(r'\\[{}]', '', line)
            brace += cleaned.count('{') - cleaned.count('}')

    for env, f, l in env_stack:
        errors.append(f'{f}:{l}: \\begin{{{env}}} never closed')
    if brace != 0:
        errors.append(f'brace imbalance: net {{ minus }} = {brace} (nonzero)')
    for k in sorted(set(dup_labels)):
        errors.append(f'duplicate \\label{{{k}}}')

    known = bib_keys(root_path)
    for k in sorted(set(refs)):
        if k not in labels:
            errors.append(f'undefined reference: \\ref/\\autoref{{{k}}} has no \\label')
    for k in sorted(set(cites)):
        if known and k not in known:
            errors.append(f'undefined citation: \\cite{{{k}}} not in any .bib')
    for k in sorted(labels):
        if k not in set(refs):
            warnings.append(f'unused label: {k}')

    print(f'[latex_check] files={len(chunks)} labels={len(labels)} '
          f'refs={len(set(refs))} cites={len(set(cites))} bibkeys={len(known)}')
    for w in warnings:
        print(f'  WARN  {w}')
    for e in errors:
        print(f'  ERROR {e}')
    if errors:
        print(f'[latex_check] FAIL — {len(errors)} error(s)')
        return 1
    print('[latex_check] OK — no structural errors')
    return 0


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'writing/tvcg/tvcg_manuscript.tex'
    sys.exit(check(root))
