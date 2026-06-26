"""Back-compat shim — implementation moved to `dvfopt.core.slp.lp_direct_2tri`.

The 2-tri SLP solver was promoted into the installable `dvfopt` package.
This module re-exports every public and private name from its new home so
existing research runners keep importing from the old path unchanged.
"""

from dvfopt.core.slp import lp_direct_2tri as _moved

globals().update({k: v for k, v in vars(_moved).items() if not k.startswith('__')})
