"""CPU tests for the generic GradientNormBalancer (no GPU, no real model).

A tiny Linear model + two synthetic aux losses with a known gradient-norm ratio
let us check the relocated balancer: live calibration equalises the *weighted*
aux gradient norms within the budget, a heavier-gradient aux gets a
proportionally smaller weight, fixed-weight mode skips calibration, and the
warmup multiplier scales the weights.
"""

import torch
import torch.nn as nn

from sage.architecture.custom_losses import GradientNormBalancer


def _setup(c1=1.0, c2=10.0, d=8, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(d, 1)
    net_input = torch.randn(16, d)

    def recompute(out):
        base = (out ** 2).mean()
        return base, [c1 * base, c2 * base]      # aux2 has c2/c1 x the gradient

    return model, net_input, recompute


def test_calibration_equalises_weighted_aux_within_budget():
    model, net_input, recompute = _setup(c1=1.0, c2=10.0)
    bal = GradientNormBalancer(n_aux=2, balance_target=0.33,
                               balance_every=4, balance_settle=4)
    for _ in range(5):                            # calibration fires at gstep==4
        out = model(net_input)
        bce, aux = recompute(out)
        total = bal.combine(bce, aux, model, net_input, recompute)
        assert torch.isfinite(total)

    assert bal._gstep == 5
    assert bal._weights[0] > 0 and bal._weights[1] > 0          # calibrated
    # each weighted aux gradient norm == (target / n) * B_ref  (equalised)
    target_each = (bal.balance_target / 2) * bal._last_B_ref
    for i in range(2):
        wn = bal._weights[i] * bal._last_aux_norms[i]
        assert abs(wn - target_each) <= 1e-4 * target_each + 1e-9
    # aux2 has 10x the gradient -> ~10x smaller weight
    assert abs(bal._weights[0] / bal._weights[1] - 10.0) < 0.05


def test_fixed_weight_mode_skips_calibration():
    model, net_input, recompute = _setup()
    bal = GradientNormBalancer(n_aux=2, aux_weights=[0.1, 0.2], balance_every=1,
                               balance_settle=0)
    out = model(net_input)
    bce, aux = recompute(out)
    total = bal.combine(bce, aux, model, net_input, recompute)
    assert bal._weights == [0.1, 0.2]                            # never recalibrated
    expected = bce + 0.1 * aux[0] + 0.2 * aux[1]
    assert torch.allclose(total, expected)


def test_warmup_scales_weights():
    model, net_input, recompute = _setup()
    bal = GradientNormBalancer(n_aux=2, aux_weights=[0.4, 0.6])
    out = model(net_input)
    bce, aux = recompute(out)
    bal.combine(bce, aux, model, net_input, recompute, warmup=0.5)
    assert bal._last_weights == [0.2, 0.3]                       # 0.5 * fixed weights


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL GRADIENT-BALANCER TESTS PASSED <<<")
