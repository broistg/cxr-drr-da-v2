import pytest

torch = pytest.importorskip("torch")

from drr_pipeline import lmmd as L


def test_zero_when_domains_identical():
    torch.manual_seed(0)
    f = torch.randn(16, 8)
    y = torch.tensor([0.0, 1.0] * 8)
    # target = the source samples with the same (hard) class memberships: LMMD must vanish
    assert abs(float(L.lmmd(f, f.clone(), y, y.clone()))) < 1e-4


def test_grows_with_class_conditional_shift():
    torch.manual_seed(0)
    y = torch.tensor([0.0, 1.0] * 16)
    f_s = torch.randn(32, 8) + y[:, None] * 3.0
    same = f_s + 0.05 * torch.randn_like(f_s)
    shifted = same + 2.0
    a = float(L.lmmd(f_s, same, y, y.clone()))
    b = float(L.lmmd(f_s, shifted, y, y.clone()))
    assert b > a > -1e-6


def test_wrong_target_classes_cost_more_than_right_ones():
    torch.manual_seed(1)
    y = torch.tensor([0.0, 1.0] * 16)
    f_s = torch.randn(32, 8) + y[:, None] * 3.0
    f_t = torch.randn(32, 8) + y[:, None] * 3.0
    right = float(L.lmmd(f_s, f_t, y, y.clone()))
    wrong = float(L.lmmd(f_s, f_t, y, 1 - y))                  # class memberships swapped
    assert wrong > right


def test_gradient_flows_to_features_not_to_pseudo_labels():
    f_s = torch.randn(8, 4, requires_grad=True)
    f_t = torch.randn(8, 4, requires_grad=True)
    p = torch.rand(8, requires_grad=True)
    y = torch.tensor([0.0, 1.0] * 4)
    L.lmmd(f_s, f_t, y, p).backward()
    assert f_s.grad is not None and f_t.grad is not None and torch.isfinite(f_t.grad).all()
    assert p.grad is None


def test_single_class_source_batch_is_finite():
    f = torch.randn(6, 4)
    assert torch.isfinite(L.lmmd(f, torch.randn(6, 4), torch.zeros(6), torch.rand(6)))


def test_ramp():
    assert L.ramp(0, 10, 60) == 0.0 and L.ramp(9, 10, 60) == 0.0
    assert L.ramp(10, 10, 60) == 0.0                    # p = 0: DSAN's schedule starts at exactly 0
    assert 0.0 < L.ramp(11, 10, 60) < 0.2
    assert L.ramp(11, 10, 60) < L.ramp(30, 10, 60) < L.ramp(59, 10, 60)
    assert L.ramp(60, 10, 60) > 0.999
