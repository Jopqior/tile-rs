#!/usr/bin/env python3
"""Experiment-only: replace tile_std's Cargo --extern proc-macro with Release dylib.

Cargo still builds the tagged macro crate as a dependency. This wrapper ensures the
actual tile_std rustc invocation links the published dylib instead; logs every swap.
No production source is changed.
"""
import os
import sys

args = sys.argv[1:]
if '--crate-name' in args and args[args.index('--crate-name') + 1] == 'tile_std':
    swaps = 0
    for index, arg in enumerate(args):
        if arg == '--extern' and index + 1 < len(args) and args[index + 1].startswith('tile_std_macros='):
            original = args[index + 1]
            packaged = os.environ['BUNDLE_MACRO']
            if not os.path.isfile(packaged):
                raise SystemExit('bundle macro missing: ' + packaged)
            args[index + 1] = 'tile_std_macros=' + packaged
            with open(os.environ['WRAPPER_LOG'], 'a') as log:
                log.write('tile_std --extern ' + original + ' -> ' + args[index + 1] + '\n')
            swaps += 1
    if swaps != 1:
        raise SystemExit(f'expected one tile_std macro --extern swap, got {swaps}')
os.execv(args[0], args)
