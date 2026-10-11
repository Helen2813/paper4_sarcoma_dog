#!/usr/bin/env python3
"""Separate source authorization for the exact V0 addendum. No training.
Save beside p5v0run.py in Downloads. Run from the project workspace root.
Only standard-library modules are imported; no numerical array or GPU access.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
EXPECTED_RUNNER_SHA = 'a5c2251e4cc2a892ece42941b4ffa7c99441ea19491247285e90766a6e6c3a26'


def main() -> int:
    print('Paper 5 V0 - exact-source authorizer; NO model execution',flush=True)
    try:
        if len(sys.argv)!=1:
            raise RuntimeError('No command-line switches; run from the project root')
        me=Path(__file__).absolute();runner=me.with_name('p5v0run.py');project=Path.cwd().absolute()
        if me.name!='p5v0auth.py' or not(project/'paper5/scripts').is_dir():
            raise RuntimeError('Keep p5v0auth.py in Downloads; use the workspace root as current directory')
        if not runner.is_file() or runner.stat().st_size>5*1024*1024:
            raise RuntimeError('Put the exact p5v0run.py beside this authorizer')
        raw=runner.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=EXPECTED_RUNNER_SHA:
            raise RuntimeError('Exact downloaded p5v0run.py SHA mismatch; do not modify its hash')
        name='p5v0_authorization_exact_executor'
        spec=importlib.util.spec_from_file_location(name,runner)
        if spec is None:
            raise RuntimeError('Cannot create executor source module')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:
            exec(compile(raw,str(runner),'exec',dont_inherit=True),module.__dict__)
            module.authorize(project,me,runner,EXPECTED_RUNNER_SHA)
        finally:
            sys.modules.pop(name,None)
        print('STOP. A separate p5v0run.py invocation may now start real V0 training.',flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as e:
        print('P5 V0 AUTHORIZATION STOPPED: '+type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        print('No benchmark started by this authorizer. Preserve old files and locks.',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
