"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte (reine Funktionen). Nur numpy/scipy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ord_constants as C
import ord_judgments as jd
import ord_score as sc
import ord_uta as ut
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    r: int = C.DEFAULT_R
    gamma: int = C.DEFAULT_GAMMA
    sigma: float = C.DEFAULT_SIGMA
    decider: str = C.DEFAULT_DECIDER
    seed: int = C.DEFAULT_SEED


@dataclass
class Case:
    """Ein gelöster Fall: Welt, Beispielurteile, UTA-Lösung und (wenn verträglich) die Relationen unter den Nicht-Referenzen."""
    S: np.ndarray
    G: np.ndarray
    w: np.ndarray
    V: np.ndarray
    utility: np.ndarray        # wahrer Nutzen des Entscheiders
    observed: np.ndarray       # Nutzen, nach dem er urteilt (mit Rauschen)
    refs: list
    others: list
    pairs: list                # Paare aufeinanderfolgender Referenzen (a vor b)
    Cm: np.ndarray
    eps: float
    compatible: bool
    d: np.ndarray
    u_hat: np.ndarray          # Nutzen der Lösung mit größtem Abstand, alle Alternativen
    region: object | None
    mn: np.ndarray | None      # min (u(a) - u(b)) unter den Nicht-Referenzen
    nec: np.ndarray | None
    pos: np.ndarray | None


def build_world(n: int, m: int, seed: int):
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    S = np.asarray(x.signed, dtype=float)
    G = np.asarray(x.goodness, dtype=float)
    w = np.asarray(x.weights, dtype=float)
    return S, G, w, sc.normalize_nominal(S)


def effective_r(n: int, r: int) -> int:
    """Mindestens zwei Nicht-Referenzen bleiben übrig (sonst gibt es kein Paar zu beurteilen)."""
    return int(max(0, min(r, n - 2)))


def solve_case(S, G, w, V, decider: str, r: int, gamma: int, sigma: float, seed: int, relations: bool = True) -> Case:
    n = len(V)
    r_eff = effective_r(n, r)
    refs = jd.choose_references(n, r_eff, SplitMix64(seed + C.OFFSET_REFS + r_eff))
    others = [i for i in range(n) if i not in refs]
    u = sc.decider_utility(decider, G, S, w)
    obs = jd.observed_utility(u, sigma, SplitMix64(seed + C.OFFSET_NOISE))
    pairs = jd.reference_pairs(obs, refs)
    Cm = ut.basis_coefficients(V, gamma)
    eps, d = ut.uta_maximin(Cm, pairs)
    compatible = ut.is_compatible(eps)
    region = ut.make_region(Cm, pairs) if compatible else None
    mn = nec = pos = None
    if relations and compatible:
        mn = ut.min_differences(Cm, region, others)
        if mn is not None:
            nec, pos = ut.necessary(mn), ut.possible(mn)
    return Case(S, G, w, V, u, obs, refs, others, pairs, Cm, eps, compatible, d, Cm @ d, region, mn, nec, pos)


def true_pairs(case: Case) -> list[tuple[int, int]]:
    """Paare (i, j) mit i < j (Indizes in `others`), deren wahrer Nutzen verschieden ist."""
    tu = case.utility[case.others]
    k = len(tu)
    return [(i, j) for i in range(k) for j in range(i + 1, k) if tu[i] != tu[j]]


def case_metrics(case: Case) -> dict:
    """Kennzahlen eines Falls gegen den wahren Nutzen (nur Paare mit verschiedenem wahrem Nutzen)."""
    tu = case.utility[case.others]
    pu = case.u_hat[case.others]
    k = len(tu)
    prs = true_pairs(case)
    out = {"compatible": case.compatible, "eps": case.eps, "k": k, "n_pairs": len(prs), "decided": None, "correct": None, "wrong_n": None, "undecided_acc": None, "width": None,
           "possible_best": None, "necessary_best": None, "best_possible": None, "uta_pairs": None, "tau": None, "top1": None}
    if not prs:
        return out
    out["uta_pairs"] = float(np.mean([(pu[i] - pu[j]) * (tu[i] - tu[j]) > 0 for i, j in prs]))
    out["tau"] = sc.kendall_tau_b(pu, tu)
    out["top1"] = bool(int(np.argmax(pu)) == int(np.argmax(tu)))
    if case.nec is None:
        return out
    nec = case.nec
    decided, right, undec = [], [], []
    for i, j in prs:
        if nec[i, j]:
            decided.append(1)
            right.append(int(tu[i] > tu[j]))
        elif nec[j, i]:
            decided.append(1)
            right.append(int(tu[j] > tu[i]))
        else:
            decided.append(0)
            undec.append(int((pu[i] - pu[j]) * (tu[i] - tu[j]) > 0))
    best, worst = ut.rank_interval(nec)
    out["decided"] = float(np.mean(decided))
    out["correct"] = float(np.mean(right)) if right else None
    out["wrong_n"] = int(len(right) - sum(right))
    out["undecided_acc"] = float(np.mean(undec)) if undec else None
    out["width"] = float(np.mean(worst - best))
    out["possible_best"] = int((best == 1).sum())
    out["necessary_best"] = int((nec.sum(axis=1) == k - 1).sum())
    out["best_possible"] = bool(best[int(np.argmax(tu))] == 1)
    return out


@dataclass
class Analysis:
    settings: Settings
    case: Case
    metrics: dict
    r_eff: int
    true_rank: np.ndarray      # Rang unter den Nicht-Referenzen nach dem wahren Nutzen (1 = bester)
    uta_rank: np.ndarray       # Rang nach der Lösung mit größtem Abstand
    best_rank: np.ndarray | None
    worst_rank: np.ndarray | None
    marginals: np.ndarray      # m x (gamma + 1)


def _ranks(u: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(u)), -np.asarray(u, dtype=float)))
    r = np.empty(len(u), dtype=int)
    r[order] = np.arange(1, len(u) + 1)
    return r


def analyse(settings: Settings) -> Analysis:
    s = settings
    S, G, w, V = build_world(s.n, s.m, s.seed)
    case = solve_case(S, G, w, V, s.decider, s.r, s.gamma, s.sigma, s.seed)
    best = worst = None
    if case.nec is not None:
        best, worst = ut.rank_interval(case.nec)
    return Analysis(s, case, case_metrics(case), effective_r(s.n, s.r), _ranks(case.utility[case.others]), _ranks(case.u_hat[case.others]), best, worst,
                    ut.marginals(case.d, s.m, s.gamma))


# ----------------------------------------------------------------------------- Experimente

def _mean(rows: list[dict], key: str):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def experiment_examples(rs=C.EXP_RS, gammas=C.GAMMA_CHOICES, seeds=C.SWEEP_SEEDS[: C.EXP_MATRICES], n=C.EXP_N, m=C.EXP_M) -> list[dict]:
    """Experiment 1: Wie viele Beispielurteile? Entscheider mit Rohwerten (in der Modellklasse), widerspruchsfreie Urteile."""
    out = []
    for gamma in gammas:
        for r in rs:
            rows = []
            for seed in seeds:
                S, G, w, V = build_world(n, m, seed)
                rows.append(case_metrics(solve_case(S, G, w, V, "roh_linear", r, gamma, 0.0, seed)))
            out.append({"r": r, "gamma": gamma, "n": n, "compatible": float(np.mean([x["compatible"] for x in rows])), "decided": _mean(rows, "decided"), "correct": _mean(rows, "correct"),
                        "width": _mean(rows, "width"), "possible_best": _mean(rows, "possible_best"), "necessary_best": _mean(rows, "necessary_best"),
                        "uta_pairs": _mean(rows, "uta_pairs"), "best_possible": _mean(rows, "best_possible"), "k": n - r})
    return out


def experiment_model(r=6, deciders=("roh_linear", "guete_linear", "untergrenze"), sigmas=(0.1, 0.2), gammas=C.GAMMA_CHOICES, seeds=C.SWEEP_SEEDS[: C.EXP_MATRICES],
                     n=C.EXP_N, m=C.EXP_M) -> list[dict]:
    """Experiment 2: Passt das Modell nicht oder sind die Urteile verrauscht? Je Entscheider (σ = 0) und je Rauschen (Entscheider mit Rohwerten)."""
    configs = [("decider", d, 0.0) for d in deciders] + [("noise", "roh_linear", s_) for s_ in sigmas]
    out = []
    for kind, decider, sigma in configs:
        for gamma in gammas:
            rows = []
            for seed in seeds:
                S, G, w, V = build_world(n, m, seed)
                rows.append(case_metrics(solve_case(S, G, w, V, decider, r, gamma, sigma, seed)))
            out.append({"kind": kind, "decider": decider, "sigma": sigma, "gamma": gamma, "r": r, "compatible": float(np.mean([x["compatible"] for x in rows])),
                        "decided": _mean(rows, "decided"), "correct": _mean(rows, "correct"), "wrong_n": _mean(rows, "wrong_n"), "uta_pairs": _mean(rows, "uta_pairs")})
    return out


def _disagreement(preds: list[np.ndarray], prs: list[tuple[int, int]]) -> float | None:
    vals = []
    for a in range(len(preds)):
        for b in range(a + 1, len(preds)):
            vals.append(np.mean([(preds[a][i] - preds[a][j]) * (preds[b][i] - preds[b][j]) < 0 for i, j in prs]))
    return float(np.mean(vals)) if vals else None


def experiment_arbitrary(rs=(4, 6, 8), gammas=C.GAMMA_CHOICES, seeds=C.SWEEP_SEEDS[: C.EXP_MATRICES], n=C.EXP_N, m=C.EXP_M, k_random=C.EXP_RANDOM_VERTICES) -> list[dict]:
    """Experiment 3: Wie willkürlich ist eine einzelne Lösung? Auf den nicht entschiedenen Paaren: Lösung mit größtem Abstand gegen zufällige verträgliche Ecken."""
    out = []
    for gamma in gammas:
        for r in rs:
            undec, maximin, rand, dis = [], [], [], []
            for seed in seeds:
                S, G, w, V = build_world(n, m, seed)
                case = solve_case(S, G, w, V, "roh_linear", r, gamma, 0.0, seed)
                if case.nec is None:
                    continue
                tu = case.utility[case.others]
                pu = case.u_hat[case.others]
                prs = [(i, j) for i, j in true_pairs(case) if not case.nec[i, j] and not case.nec[j, i]]
                total = len(true_pairs(case))
                undec.append(len(prs) / total if total else 0.0)
                if not prs:
                    continue
                maximin.append(np.mean([(pu[i] - pu[j]) * (tu[i] - tu[j]) > 0 for i, j in prs]))
                rng = SplitMix64(seed + C.OFFSET_VERTEX + r)
                preds = []
                for _ in range(k_random):
                    dv = ut.random_vertex(case.region, rng)
                    if dv is not None:
                        preds.append((case.Cm @ dv)[case.others])
                if preds:
                    rand.append(np.mean([np.mean([(p[i] - p[j]) * (tu[i] - tu[j]) > 0 for i, j in prs]) for p in preds]))
                    dd = _disagreement(preds, prs)
                    if dd is not None:
                        dis.append(dd)
            out.append({"r": r, "gamma": gamma, "undecided": float(np.mean(undec)) if undec else None, "maximin": float(np.mean(maximin)) if maximin else None,
                        "random": float(np.mean(rand)) if rand else None, "disagree": float(np.mean(dis)) if dis else None})
    return out


def experiment_sampling(rs=(3, 4, 6, 8), seeds=C.SWEEP_SEEDS[: C.EXP_SAMPLING_MATRICES], n=C.EXP_N, m=C.EXP_M, draws=C.EXP_DRAWS) -> list[dict]:
    """Experiment 4: Gesampelt gegen ausgelotet (nur Gewichte, gamma = 1): Anteil der gleichverteilten Gewichtsziehungen, die die Ordnung der Referenzen erfüllen,
    und die möglichen Beste in der Stichprobe gegen den LP."""
    out = []
    for r in rs:
        share, acc_n, lp_n, found_n, miss_n, none, in_region = [], [], [], [], [], 0, 0
        for seed in seeds:
            S, G, w, V = build_world(n, m, seed)
            case = solve_case(S, G, w, V, "roh_linear", r, 1, 0.0, seed, relations=False)
            g = -np.log1p(-SplitMix64(seed + C.OFFSET_DRAWS).uniform(draws * m)).reshape(draws, m)
            W = g / g.sum(axis=1, keepdims=True)
            ok = ut.share_compatible(V, case.pairs, W)
            acc = W[ok]
            share.append(float(ok.mean()))
            acc_n.append(len(acc))
            poss = ut.possible_best_in_region(V, case.pairs)
            lp_n.append(int(poss.sum()))
            in_region += int(poss[int(np.argmax(case.utility))])
            if len(acc) == 0:
                none += 1
                continue
            found = np.bincount((acc @ V.T).argmax(axis=1), minlength=n) > 0
            found_n.append(int(found.sum()))
            miss_n.append(int((poss & ~found).sum()))
        out.append({"r": r, "share": float(np.mean(share)), "accepted": float(np.mean(acc_n)), "none": none, "possible_lp": float(np.mean(lp_n)),
                    "found": float(np.mean(found_n)) if found_n else None, "missed": float(np.mean(miss_n)) if miss_n else None, "in_region": in_region, "matrices": len(seeds),
                    "draws": draws})
    return out


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def _row(rows: list[dict], **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def verdict_examples(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    lo, hi = min(r["r"] for r in rows), max(r["r"] for r in rows)
    a1, a2 = _row(rows, r=lo, gamma=1), _row(rows, r=hi, gamma=1)
    b1, b2 = _row(rows, r=lo, gamma=3), _row(rows, r=hi, gamma=3)
    text = (f"Liegt der Nutzen des Entscheiders in der Modellklasse, sind alle entschiedenen Paare richtig ({pct(a2['correct'])} %). Entschieden sind bei {lo} Beispielurteilen von {a1['n']} "
            f"Alternativen {pct(a1['decided'])} % der übrigen Paare, bei {hi} sind es {pct(a2['decided'])} %; mit stückweise linearen Randfunktionen {pct(b1['decided'])} % und {pct(b2['decided'])} %. "
            f"Die Rangbreite sinkt von {de(a1['width'], 1)} auf {de(a2['width'], 1)} Plätze.")
    return "info", text


def verdict_model(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    ok = _row(rows, kind="decider", decider="roh_linear", gamma=1)
    nl = _row(rows, kind="decider", decider="untergrenze", gamma=1)
    noisy = _row(rows, kind="noise", sigma=0.2, gamma=1)
    text = (f"Passen Modell und Urteile, sind {pct(ok['correct'])} % der entschiedenen Paare richtig (Ordnung verträglich in {pct(ok['compatible'])} % der Matrizen). Beim nicht additiven "
            f"Entscheider (Untergrenze) sind {pct(1 - nl['correct'])} % der „notwendigen“ Paare falsch. Bei Urteilen mit 20 % Rauschen ist die Ordnung der Referenzen nur in "
            f"{pct(noisy['compatible'])} % der Matrizen verträglich, und {pct(1 - noisy['correct'])} % der entschiedenen Paare (unter den verträglichen Matrizen) sind falsch: „notwendig“ gilt nur unter Voraussetzungen.")
    return "warning", text


def verdict_arbitrary(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    mid = _row(rows, r=6, gamma=1)
    flex = _row(rows, r=6, gamma=3)
    text = (f"Bei 6 Beispielurteilen bleiben {pct(mid['undecided'])} % der Paare offen. Dort ordnet die Lösung mit größtem Abstand {pct(mid['maximin'])} % richtig, eine zufällige verträgliche "
            f"Ecke {pct(mid['random'])} %, und zwei zufällige Ecken ordnen {pct(mid['disagree'])} % der offenen Paare verschieden (mit stückweise linearen Randfunktionen "
            f"{pct(flex['disagree'])} %): Eine einzelne Lösung ist eine Wahl, keine Antwort.")
    return "info", text


def verdict_sampling(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 2) if x < 0.01 else de(100 * x, 1)  # noqa: E731
    lo, hi = min(r["r"] for r in rows), max(r["r"] for r in rows)
    a, b = _row(rows, r=lo), _row(rows, r=hi)
    text = (f"Von {de(a['draws'], 0)} gleichverteilten Gewichtsziehungen erfüllen bei {lo} Referenzen {pct(a['share'])} % die Ordnung, bei {hi} nur {pct(b['share'])} % "
            f"(im Mittel {de(b['accepted'], 0)} Ziehungen). Der LP findet {de(b['possible_lp'], 2)} mögliche Beste je Matrix, die Stichprobe {de(b['found'], 2)}; bei {lo} sind es "
            f"{de(a['possible_lp'], 2)} gegen {de(a['found'], 2)}. Mit jedem Urteil trifft die Ziehung seltener, der LP braucht je Alternative weiter eine Rechnung.")
    return "info", text
