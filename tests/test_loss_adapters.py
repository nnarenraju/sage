"""CPU tests for the LossAdapter protocol + SageVanillaTraining._collect.

Two fake adapters (a main loss with a primary + aux components, and an aux-only
loss) verify that the trainer's multi-loss collection pulls out exactly one
primary term, the flat list of aux terms in order, and each adapter's own total
for logging. ``_collect`` is exercised directly via ``__new__`` so no GPU /
config / samplers are needed.
"""

import torch

from sage.factory.loss_adapters import LossAdapter
from sage.factory.training import SageVanillaTraining


class _MainAdapter(LossAdapter):
    primary_index = 1            # bce
    aux_indices = (2, 3)         # pe_reg, coupling

    def components(self, out, targets, ctx):
        return torch.tensor([10.0, 1.0, 2.0, 3.0])   # [total, bce, pe_reg, coupling]


class _AuxAdapter(LossAdapter):
    primary_index = None         # contributes only aux
    aux_indices = (1, 2)         # cons_tc, cons_mc

    def components(self, out, targets, ctx):
        return torch.tensor([5.0, 0.5, 0.6])         # [total, cons_tc, cons_mc]


def _bare_trainer(main, aux):
    t = SageVanillaTraining.__new__(SageVanillaTraining)   # skip heavy __init__
    t.loss_function = main
    t.aux_losses = list(aux)
    return t


def _r(xs):
    return [round(float(x), 4) for x in xs]   # float32 -> tidy compare


def test_collect_pulls_primary_aux_and_totals():
    t = _bare_trainer(_MainAdapter(), [_AuxAdapter()])
    primary, aux_terms, totals = t._collect(None, None, {})
    assert round(float(primary), 4) == 1.0                # the BCE term
    assert _r(aux_terms) == [2.0, 3.0, 0.5, 0.6]          # in adapter order
    assert _r(totals) == [10.0, 5.0]                      # main, aux totals


def test_collect_main_only():
    t = _bare_trainer(_MainAdapter(), [])
    primary, aux_terms, totals = t._collect(None, None, {})
    assert round(float(primary), 4) == 1.0
    assert _r(aux_terms) == [2.0, 3.0]
    assert _r(totals) == [10.0]


def test_adapter_is_callable():
    a = _MainAdapter()
    assert torch.equal(a(None, None, {}), a.components(None, None, {}))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL LOSS-ADAPTER TESTS PASSED <<<")
