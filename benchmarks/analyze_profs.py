"""Offline analysis of the crop-benchmark .prof dumps: module-level accounting.

Groups cumulative/self time by module so the pass total is fully attributed
(the console top-N view truncated ~70% of the sliver no-TR pass), then prints
the top self-time functions and the OSQP call picture.
"""

import glob
import os
import pstats

OUT = "benchmarks/output/isqp_campaign"


def module_of(key):
    path = key[0]
    if "site-packages" in path:
        tail = path.split("site-packages")[-1].lstrip("\\/")
        return "pkg:" + tail.split("\\")[0].split("/")[0]
    if "dvfopt" in path:
        tail = path.split("deformation-field-processing")[-1].lstrip("\\/")
        return tail.rsplit(".py", 1)[0]
    if path.startswith(("<", "~")):
        return "builtin/C"
    return "python-stdlib"


for prof in sorted(glob.glob(f"{OUT}/*.prof")):
    st = pstats.Stats(prof)
    total = st.total_tt
    agg = {}
    for key, (_cc, _nc, tt, _ct, _callers) in st.stats.items():
        m = module_of(key)
        agg[m] = agg.get(m, 0.0) + tt
    print(f"\n===== {os.path.basename(prof)}  total={total:.1f}s")
    for m, tt in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {tt:8.1f}s {100 * tt / total:5.1f}%  {m}")
    # top self-time functions
    rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:8]
    print("  -- top self-time functions:")
    for (path, line, fn), (_cc, nc, tt, _ct, _callers) in rows:
        short = path.split("\\")[-1].split("/")[-1]
        print(f"  {tt:8.1f}s n={nc:<9} {short}:{line}({fn})")
