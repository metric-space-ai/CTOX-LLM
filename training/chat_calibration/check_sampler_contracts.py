#!/usr/bin/env python3
"""Verify sampler parity and prove the factor-of-two TV contract detects a mutation."""
import io,unittest
from pathlib import Path
import fit
import test_sampler_objective as contract
suite=unittest.defaultTestLoader.loadTestsFromModule(contract)
if not unittest.TextTestRunner().run(suite).wasSuccessful():raise SystemExit(1)
source=Path(fit.__file__).read_text()
old='return .5*(q-reference.double()).abs().sum(-1).mean()'
if source.count(old)!=1:raise SystemExit('mutation anchor differs')
namespace=dict(fit.__dict__);exec(compile(source.replace(old,old.replace('.5*','1.*')),fit.__file__,'exec'),namespace)
contract.sampler_tv_loss=namespace['sampler_tv_loss']
r=unittest.TextTestRunner(stream=io.StringIO()).run(contract.SamplerObjective('test_value_and_gradient'))
if r.wasSuccessful() or r.errors or len(r.failures)!=1:raise SystemExit('factor-of-two mutation not detected')
print('MUTATION PROOF: TV factor .5 -> 1; test_value_and_gradient fails')
