from src.eval.report import ComparisonReport


def _rep():
    r = ComparisonReport(split_regime="donor_held_out", group_key="donor_id",
                         is_synthetic=True, provenance="synthetic",
                         n_train=100, n_test=50, n_test_groups=4)
    return r


def test_verdict_reports_a_loss_plainly():
    r = _rep()
    r.add("foundation_model", {"probe_macro_f1": 0.40})
    r.add("baseline:pca", {"probe_macro_f1": 0.80})
    assert "does NOT beat" in r.verdict()


def test_verdict_requires_ci_separation_for_a_win():
    r = _rep()
    r.add("foundation_model", {"probe_macro_f1": 0.81, "probe_macro_f1_lo": 0.70})
    r.add("baseline:pca", {"probe_macro_f1": 0.80})
    assert "not a demonstrated improvement" in r.verdict()


def test_verdict_accepts_a_clean_win():
    r = _rep()
    r.add("foundation_model", {"probe_macro_f1": 0.95, "probe_macro_f1_lo": 0.92})
    r.add("baseline:pca", {"probe_macro_f1": 0.80})
    assert "beats best baseline" in r.verdict()


def test_no_baselines_gives_no_verdict():
    r = _rep()
    r.add("foundation_model", {"probe_macro_f1": 0.9})
    assert "no verdict" in r.verdict()


def test_text_flags_synthetic_data_and_hides_ci_columns():
    r = _rep()
    r.add("foundation_model", {"probe_macro_f1": 0.5, "probe_macro_f1_lo": 0.4})
    t = r.to_text()
    assert "SYNTHETIC" in t
    assert "probe_macro_f1_lo" not in t


def test_lower_is_better_columns_are_marked_correctly():
    r = _rep()
    r.add("a", {"pcr_batch": 0.1})
    r.add("b", {"pcr_batch": 0.5})
    lines = {l.split()[0]: l for l in r.to_text().splitlines()
             if l.strip() and l.split()[0] in ("a", "b")}
    assert "*" in lines["a"] and "*" not in lines["b"].split()[1]
