"""Back-compat shim — implementation moved to `dvfopt.core.slp.tri_linearize`.

The 2-tri SLP solver was promoted into the installable `dvfopt` package.
This module re-exports every public and private name from its new home so
existing research runners keep importing from the old path unchanged.
"""

from dvfopt.core.slp import tri_linearize as _moved

globals().update({k: v for k, v in vars(_moved).items() if not k.startswith('__')})
