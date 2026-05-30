from fraud.report import render_report


def test_render_report_contains_dollars(tmp_path):
    results = {
        "net_saved": 123456.0, "threshold": 0.42, "recall": 0.85,
        "precision": 0.10, "fraud_caught_pct": 0.83, "figures": [], "shap_top": [],
    }
    out = tmp_path / "report.html"
    render_report(results, out)
    html = out.read_text()
    assert "123,456" in html and "0.42" in html
