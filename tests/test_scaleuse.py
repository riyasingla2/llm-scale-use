import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from scaleuse import analysis as A
from scaleuse import probe as P
from scaleuse import run_blackbox
from scaleuse.backends import parse_rating
from scaleuse.items import FORMATS, VIGNETTES


class _OpenAIHandler(BaseHTTPRequestHandler):
    """Minimal fake OpenAI chat-completions endpoint; always returns "7"."""

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = json.dumps({"choices": [{"message": {"content": "7"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


@pytest.fixture(scope="module")
def fake_openai_url():
    srv = HTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1/chat/completions"
    srv.shutdown()


def test_parse_rating():
    assert parse_rating("7", 0, 10) == 7
    assert parse_rating("I'd say 8.", 0, 10) == 8
    assert parse_rating("11", 0, 10) is None
    assert parse_rating("no idea", 0, 10) is None
    assert parse_rating("", 1, 5) is None


def test_parse_rating_strips_think():
    # number inside <think> must not be returned; number after must be
    assert parse_rating("<think>maybe 11</think>7", 0, 10) == 7
    # out-of-range number inside think, in-range number after
    assert parse_rating("<think>answer is 5</think>3", 1, 5) == 3
    # multiline think block
    assert parse_rating("<think>\nlong reasoning\n11\n</think>4", 0, 10) == 4


def test_chat_api_backend_end_to_end(fake_openai_url):
    from scaleuse.backends import ChatAPIBackend
    backend = ChatAPIBackend("openrouter", api_key="test-key", url=fake_openai_url)
    result = backend.chat("some-model", "system prompt", "user prompt", temperature=1.0, seed=0)
    assert result == "7"


def test_design_is_balanced():
    levels = pd.Series([v.level for v in VIGNETTES]).value_counts()
    assert set(levels.index) == {1, 2, 3, 4, 5} and (levels == 4).all()
    domains = pd.Series([v.domain for v in VIGNETTES]).value_counts()
    assert (domains == 5).all()


def test_icc1_known_values():
    perfect = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3.0]])
    assert A.icc1(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    pure_noise = rng.normal(size=(200, 5))
    assert abs(A.icc1(pure_noise)) < 0.05


def test_departure_linear_vs_bent():
    lin = pd.Series([0, .25, .5, .75, 1], index=range(1, 6))
    bent = pd.Series([0, .5, .8, .95, 1], index=range(1, 6))
    assert A.departure_from_equal_spacing(lin) == pytest.approx(0)
    assert A.departure_from_equal_spacing(bent) > 0.15
    # affine transformations of the scale must not count as departures
    assert A.departure_from_equal_spacing(lin * 0.5 + 0.2) == pytest.approx(0)


def test_lack_of_fit_deterministic():
    # perfectly linear deterministic: both lin and cat fit perfectly -> 1.0
    linear = pd.DataFrame({
        "level": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "norm":  [0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0],
    })
    assert A.lack_of_fit_p(linear) == pytest.approx(1.0)
    # perfectly bent deterministic: cat fits perfectly but lin doesn't -> 0.0
    bent = pd.DataFrame({
        "level": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "norm":  [0.0, 0.0, 0.6, 0.6, 0.85, 0.85, 0.95, 0.95, 1.0, 1.0],
    })
    assert A.lack_of_fit_p(bent) == pytest.approx(0.0)


def test_king_wand_c():
    anchors = np.array([0.1, 0.3, 0.5])
    assert A.king_wand_c(0.0, anchors) == (1, True)
    assert A.king_wand_c(0.1, anchors) == (2, True)
    assert A.king_wand_c(0.2, anchors) == (3, True)
    assert A.king_wand_c(0.5, anchors) == (6, True)
    assert A.king_wand_c(0.9, anchors) == (7, True)
    assert A.king_wand_c(0.2, np.array([0.3, 0.1, 0.5]))[1] is False


def test_vignette_adjustment_removes_scale_use_differences():
    """Two respondents with the same true state but different scale use get
    different raw self-ratings but the same vignette-adjusted rating."""
    lin = pd.Series([0, .25, .5, .75, 1], index=range(1, 6))
    opt = lin ** 0.5  # inflates the low end
    true_level = 3.5
    raw_lin = np.interp(true_level, lin.index, lin.values)
    raw_opt = np.interp(true_level, opt.index, opt.values)
    assert raw_opt - raw_lin > 0.05
    assert A.vignette_adjusted(raw_lin, lin) == pytest.approx(true_level)
    assert A.vignette_adjusted(raw_opt, opt) == pytest.approx(true_level)


def test_mock_pipeline_recovers_known_scale_use(tmp_path):
    out = tmp_path / "r.jsonl"
    run_blackbox.main(["--backend", "mock", "--reps", "3", "--out", str(out)])
    # resume: second run makes no new calls
    n_lines = len(out.read_text().splitlines())
    run_blackbox.main(["--backend", "mock", "--reps", "3", "--out", str(out)])
    assert len(out.read_text().splitlines()) == n_lines

    summary, _ = A.summarise(A.load(out))
    s = summary[summary.fmt == "0-10"].set_index("model")
    assert s.loc["mock/linear", "departure"] < 0.05
    assert s.loc["mock/optimist", "departure"] > 0.1
    assert s.loc["mock/compressed", "departure"] > 0.1
    assert (s["spearman"] > 0.9).all()
    # same latent self state -> raw self-reports spread out, adjusted ones close together
    assert s["self_raw"].max() - s["self_raw"].min() > 0.2
    assert s["self_adjusted"].max() - s["self_adjusted"].min() < 0.6


def test_probe_finds_planted_direction():
    rng = np.random.default_rng(0)
    n, L, d = 40, 4, 64
    y = np.repeat(np.arange(1, 6), 8).astype(float)
    groups = np.tile(np.array(["a", "b", "c", "d"]), 10)
    X = rng.normal(size=(n, L, d))
    direction = rng.normal(size=d)
    X[:, 2, :] += np.outer(y - y.mean(), direction) * 0.5  # signal only in layer 2
    res = P.layerwise_probe(X, y, groups)
    assert res["oof_spearman"].idxmax() == 2
    assert res.loc[2, "oof_spearman"] > 0.9
    assert res.loc[0, "oof_spearman"] < 0.5


def test_permutation_p_accounts_for_selection():
    rng = np.random.default_rng(7)
    n, L, d = 40, 3, 32
    y = np.repeat(np.arange(1, 6), 8).astype(float)
    groups = np.tile(np.array(["a", "b", "c", "d"]), 10)

    # strong signal in layer 1: observed best rho unreachable by permutations
    X_sig = rng.normal(size=(n, L, d))
    X_sig[:, 1, :] += np.outer(y - y.mean(), rng.normal(size=d)) * 2.0
    res_sig = P.layerwise_probe(X_sig, y, groups)
    p_val, count = P.permutation_p(X_sig, y, groups, res_sig["oof_spearman"].max(), n_perm=200, seed=0)
    assert p_val < 0.05

    # pure noise: observed best is itself drawn from the null -> p should be large
    X_noise = rng.normal(size=(n, L, d))
    res_noise = P.layerwise_probe(X_noise, y, groups)
    p_noise, _ = P.permutation_p(X_noise, y, groups, res_noise["oof_spearman"].max(), n_perm=200, seed=0)
    assert p_noise > 0.05


def test_whitebox_prompts_cover_design():
    from scaleuse.whitebox import build_prompts
    ps = build_prompts(["0-10", "1-5"])
    assert len(ps) == 2 * (len(VIGNETTES) + 1)
    assert all(str(FORMATS[p["fmt"]].hi) in p["text"] for p in ps)


def test_whitebox_summarise_prompts_cover_design():
    from scaleuse.whitebox import build_summarise_prompts
    ps = build_summarise_prompts()
    assert len(ps) == len(VIGNETTES)          # no self item
    assert all(p["fmt"] == "summarise" for p in ps)
    assert all("Summarise" in p["text"] for p in ps)
    # no rating scale in the prompt
    assert all("scale from" not in p["text"] for p in ps)


def test_logit_vignette_adjusted_self(tmp_path):
    from scaleuse.probe import logit_vignette_adjusted_self

    # build a synthetic logit ratings file with perfect linear scale use
    rows = []
    for v in VIGNETTES:
        norm = (v.level - 1) / 4.0
        rows.append({"item_id": v.vid, "level": v.level, "domain": v.domain,
                     "fmt": "0-10", "expected": norm * 10, "expected_norm": norm})
    # self item at norm=0.75 -> level 4 on a linear profile
    rows.append({"item_id": "self", "level": -1, "domain": "self",
                 "fmt": "0-10", "expected": 7.5, "expected_norm": 0.75})

    path = tmp_path / "wb_model_logit_ratings.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = logit_vignette_adjusted_self(path, "0-10")
    assert result == pytest.approx(4.0)

    # missing file -> None
    assert logit_vignette_adjusted_self(tmp_path / "missing.jsonl", "0-10") is None


def test_probe_runs_on_summarise_npz(tmp_path, capsys):
    """A summarise-style npz (fmt='summarise', no self item) runs end-to-end."""
    rng = np.random.default_rng(3)
    n, L, d = 20, 4, 16
    y_arr = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5], dtype=float)
    domains = np.tile(["work", "health", "relationships", "money"], 5)
    direction = rng.normal(size=d)
    states = rng.normal(size=(n, L, d)).astype(np.float16)
    states[:, 2, :] = (states[:, 2, :].astype(np.float64) +
                       np.outer(y_arr - y_arr.mean(), direction * 2.0)).astype(np.float16)
    meta = [{"item_id": f"v{i+1:02d}", "level": int(y_arr[i]),
             "domain": domains[i], "fmt": "summarise"} for i in range(n)]
    npz_path = tmp_path / "wb_testmodel_summarise.npz"
    np.savez_compressed(npz_path, states=states, meta=json.dumps(meta), model="testmodel")

    from scaleuse import probe as P_mod
    P_mod.main([str(npz_path), "--n-perm", "20"])
    out = capsys.readouterr().out
    assert "best layer 2" in out
    assert "no self item" in out


def test_transfer_probe():
    rng = np.random.default_rng(0)
    n, L, d = 20, 3, 16
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], dtype=float)
    groups = np.tile(np.array(["a", "b", "c", "d"]), 5)
    direction = rng.normal(size=d)

    # same valence direction planted in both tasks -> transfer should recover it
    X_rate = rng.normal(size=(n, L, d))
    X_rate[:, 1, :] += np.outer(y - y.mean(), direction) * 2.0
    X_sum = rng.normal(size=(n, L, d))
    X_sum[:, 1, :] += np.outer(y - y.mean(), direction) * 2.0
    res = P.transfer_probe(X_rate, y, groups, X_sum)
    assert res["spearman"].idxmax() == 1
    assert res.loc[1, "spearman"] > 0.7

    # independent noise -> poor transfer
    X_noise_rate = rng.normal(size=(n, L, d))
    X_noise_sum = rng.normal(size=(n, L, d))
    res_noise = P.transfer_probe(X_noise_rate, y, groups, X_noise_sum)
    assert res_noise["spearman"].max() < 0.6
