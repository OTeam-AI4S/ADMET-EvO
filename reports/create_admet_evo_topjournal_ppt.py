#!/usr/bin/env python3
"""Build a three-slide, evidence-backed ADMET-EvO scientific overview deck."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pptx import Presentation
from pptx.util import Inches


ROOT = Path("/home/wyilin/admet")
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "ppt_assets_topjournal"
OUTPUT = REPORTS / "ADMET_EvO_TopJournal_4slide_Overview_EN.pptx"

SCORECARD = REPORTS / "goal_scorecard_v27.json"
ROUTER_CONTROLS = ROOT / "runs/v27-router-controls/report.json"
RISK_GUARD = ROOT / "runs/v27-router-risk-guard/report.json"
FAULT_SUITE = REPORTS / "v3_hidden_fault_suite.json"
POLICY_COMPARISON = REPORTS / "v3_policy_comparison_agent_strong_fixed.json"

NAVY = "#17324D"
NAVY_2 = "#244E6B"
TEAL = "#287E8E"
TEAL_DARK = "#145B67"
TEAL_LIGHT = "#DDEDEF"
GOLD = "#D5A13F"
GOLD_LIGHT = "#F6EEDC"
RED = "#A9443F"
RED_LIGHT = "#F5E5E3"
INK = "#17212B"
MUTED = "#5C6973"
GRID = "#D9E0E3"
PALE = "#F4F7F7"
WHITE = "#FFFFFF"
GREEN = "#2E7D5B"
GREEN_LIGHT = "#E2F0E9"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canvas():
    fig = plt.figure(figsize=(13.333, 7.5), dpi=180, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=12, color=INK, weight="normal", ha="left", va="center",
         family="DejaVu Sans", linespacing=1.15, zorder=10):
    return ax.text(
        x, y, value, fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, family=family, linespacing=linespacing, zorder=zorder,
    )


def box(ax, x, y, w, h, face=WHITE, edge=GRID, radius=0.012, lw=1.0, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def rule(ax, x0, y0, x1, y1, color=GRID, lw=1.0, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, linestyle=ls, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=TEAL, lw=1.6, style="-|>", rad=0.0, zorder=6):
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def header(ax, kicker, title_value, subtitle, page):
    text(ax, 0.055, 0.955, kicker.upper(), 8.2, TEAL_DARK, "bold", va="top")
    text(ax, 0.055, 0.908, title_value, 21.5, NAVY, "bold", va="top")
    text(ax, 0.055, 0.846, subtitle, 9.3, MUTED, va="top")
    text(ax, 0.945, 0.945, f"0{page}", 10, TEAL, "bold", ha="right", va="top")
    rule(ax, 0.055, 0.815, 0.945, 0.815, GRID, 0.8)


def footer(ax, source, note=""):
    rule(ax, 0.055, 0.047, 0.945, 0.047, GRID, 0.7)
    text(ax, 0.055, 0.026, source, 5.6, MUTED, va="center")
    if note:
        text(ax, 0.945, 0.026, note, 5.6, MUTED, ha="right", va="center")


def slide_architecture(out: Path):
    fig, ax = canvas()
    header(
        ax,
        "Architecture | ADMET-EvO v3",
        "A protocol-safe research agent for endpoint-adaptive model discovery",
        "Scientific hypotheses are converted into bounded actions; only frozen recipes reach the official evaluator.",
        1,
    )

    # Discovery band.
    text(ax, 0.057, 0.775, "DISCOVERY PLANE", 7.5, TEAL_DARK, "bold", va="top")
    text(ax, 0.943, 0.775, "TRAIN + VALIDATION ONLY", 7.5, TEAL_DARK, "bold", ha="right", va="top")

    columns = [
        (0.055, 0.660, 0.153, 0.102, "01", "Endpoint contract", "Semantics · metric\nassay · split ownership"),
        (0.225, 0.660, 0.153, 0.102, "02", "Hardness & faults", "Scarcity · imbalance · OOD\nnoise · overlap · cliffs"),
        (0.395, 0.660, 0.153, 0.102, "03", "Action planner", "Feature · model · data\nreplicate · abstain / stop"),
        (0.565, 0.660, 0.153, 0.102, "04", "Axis-isolated executor", "Registered features & models\nmatched controls · budget"),
        (0.735, 0.660, 0.208, 0.102, "05", "Evidence router", "Multi-seed score · uncertainty · cost\nfull-outcome memory · ledger"),
    ]
    for x, y, w, h, number, title_value, body in columns:
        box(ax, x, y, w, h, WHITE, GRID, radius=0.010, lw=0.9)
        box(ax, x + 0.009, y + h - 0.031, 0.028, 0.021, TEAL_LIGHT, TEAL_LIGHT, radius=0.004, lw=0)
        text(ax, x + 0.023, y + h - 0.020, number, 6.5, TEAL_DARK, "bold", ha="center")
        text(ax, x + 0.012, y + 0.052, title_value, 9.2, NAVY, "bold", va="center")
        text(ax, x + 0.012, y + 0.020, body, 6.7, MUTED, va="center", linespacing=1.25)
    for idx in range(len(columns) - 1):
        left = columns[idx]
        right = columns[idx + 1]
        arrow(ax, left[0] + left[2] + 0.004, 0.711, right[0] - 0.004, 0.711, NAVY_2, 1.1)

    # Research reasoning and implementation layers.
    bands = [
        (0.055, 0.548, 0.888, 0.072, TEAL_LIGHT, TEAL_DARK,
         "SCIENTIFIC REASONING", "Problem → hypothesis → falsification test → observed effect → revised claim"),
        (0.055, 0.458, 0.888, 0.072, PALE, NAVY,
         "EXECUTABLE LINEAGE", "ResearchAction → controlled configuration → training / prediction → artifact hash → reproducible snapshot"),
        (0.055, 0.368, 0.888, 0.072, GOLD_LIGHT, "#8A641F",
         "EVIDENCE LINEAGE", "Multi-seed validation → stability → promotion / rejection → immutable finalist → supported claim"),
    ]
    for x, y, w, h, face, accent, label, body in bands:
        box(ax, x, y, w, h, face, face, radius=0.008, lw=0)
        text(ax, x + 0.014, y + h / 2, label, 7.1, accent, "bold")
        text(ax, x + 0.205, y + h / 2, body, 8.1, INK)

    # Feedback loop and freeze boundary.
    arrow(ax, 0.835, 0.653, 0.470, 0.631, TEAL, 1.4, rad=-0.22)
    text(ax, 0.651, 0.607, "validation-side learning only", 6.3, TEAL_DARK, "bold", ha="center")

    rule(ax, 0.055, 0.321, 0.943, 0.321, RED, 1.1, "--")
    box(ax, 0.382, 0.293, 0.236, 0.052, NAVY, NAVY, radius=0.012, lw=0)
    text(ax, 0.500, 0.319, "SHA-256 FREEZE LOCK", 9.5, WHITE, "bold", ha="center")
    text(ax, 0.942, 0.302, "NO TEST FEEDBACK TO DISCOVERY", 6.8, RED, "bold", ha="right")

    text(ax, 0.057, 0.260, "CERTIFICATION PLANE", 7.5, NAVY, "bold", va="top")
    cert = [
        (0.070, 0.129, 0.175, "Frozen recipe", "Model · representation\nweights · protocol hash"),
        (0.292, 0.129, 0.175, "Independent retraining", "≥5 seeds\ntrain + validation"),
        (0.514, 0.129, 0.175, "Official evaluator", "Read-only test access\nonce per lineage"),
        (0.736, 0.129, 0.190, "Audit bundle", "Scores · predictions\nenvironment · provenance"),
    ]
    for x, y, w, title_value, body in cert:
        box(ax, x, y, w, 0.095, NAVY, NAVY, radius=0.010, lw=0)
        text(ax, x + w / 2, y + 0.061, title_value, 8.8, WHITE, "bold", ha="center")
        text(ax, x + w / 2, y + 0.027, body, 6.5, "#D7E4EC", ha="center", linespacing=1.2)
    for idx in range(len(cert) - 1):
        arrow(ax, cert[idx][0] + cert[idx][2] + 0.008, 0.177, cert[idx + 1][0] - 0.008, 0.177, GOLD, 1.3)

    box(ax, 0.055, 0.070, 0.888, 0.038, GREEN_LIGHT, GREEN_LIGHT, radius=0.006, lw=0)
    text(ax, 0.499, 0.089, "Design principle  |  The planner proposes; the deterministic executor, evidence router and protocol firewall decide.",
         7.5, GREEN, "bold", ha="center")

    footer(
        ax,
        "Sources: admet_agent/runner.py; research_loop.py; freeze_audit.py; docs/tdc22_research_agent_v3.md",
        "TDC-22 scope · discovery/test separation",
    )
    fig.savefig(out, dpi=180, facecolor=WHITE)
    plt.close(fig)



def slide_architecture_full(out: Path):
    """Render the complete Markdown research loop as a large swimlane flowchart."""
    fig, ax = canvas()
    header(
        ax,
        "Complete workflow | ADMET-EvO v3",
        "From endpoint diagnosis to frozen, one-shot certification",
        "A single evidence loop links scientific reasoning, bounded execution and an immutable test boundary.",
        4,
    )

    # Three full-width swimlanes.
    lanes = [
        (0.055, 0.635, 0.888, 0.145, TEAL_LIGHT, "CONTEXT & DIAGNOSIS", TEAL_DARK),
        (0.055, 0.345, 0.888, 0.270, PALE, "RESEARCH LOOP", NAVY),
        (0.055, 0.082, 0.888, 0.235, GOLD_LIGHT, "FREEZE & CERTIFICATION", "#805B1A"),
    ]
    for x, y, w, h, face, label, accent in lanes:
        box(ax, x, y, w, h, face, face, radius=0.008, lw=0)
        text(ax, x + 0.010, y + h - 0.012, label, 6.2, accent, "bold", va="top")

    def node(x, y, w, h, title_value, body, face=WHITE, edge=GRID, accent=NAVY, title_size=7.2, body_size=5.4):
        box(ax, x, y, w, h, face, edge, radius=0.007, lw=0.8)
        text(ax, x + w / 2, y + h * 0.64, title_value, title_size, accent, "bold", ha="center")
        text(ax, x + w / 2, y + h * 0.29, body, body_size, MUTED if face == WHITE else INK,
             ha="center", linespacing=1.18)

    # Full diagnostic chain from the Markdown architecture.
    diagnosis = [
        (0.075, "Raw endpoint data", "Structures · labels\nsource · assay metadata"),
        (0.217, "Endpoint contract", "Property · species · unit\nmetric · split ownership"),
        (0.359, "Protocol guard", "Freeze split · metric\ntest rights · budget"),
        (0.501, "Hardness profiler", "Scarcity · imbalance · OOD\nroughness · noise · overlap"),
        (0.643, "Problem diagnostician", "Root cause · evidence\nfalsification test"),
        (0.785, "Action router", "Select the next\nresearch action"),
    ]
    for x, title_value, body in diagnosis:
        node(x, 0.665, 0.124, 0.081, title_value, body, WHITE, GRID, NAVY, 6.7, 4.8)
    for i in range(len(diagnosis) - 1):
        arrow(ax, diagnosis[i][0] + 0.126, 0.706, diagnosis[i + 1][0] - 0.003, 0.706, NAVY_2, 1.0)

    # Five single-axis research actions.
    actions = [
        (0.075, "FEATURE", "Descriptors · SMARTS\npretrained views", TEAL_LIGHT, TEAL_DARK),
        (0.244, "MODEL", "Family · loss · transform\nensemble", "#E6ECF2", NAVY_2),
        (0.413, "DATA", "Provenance · assay alignment\nquarantine", GREEN_LIGHT, GREEN),
        (0.582, "REPLICATE", "Multi-seed · resampling\nstability review", GOLD_LIGHT, "#805B1A"),
        (0.751, "ABSTAIN / STOP", "Reject prediction · request data\nstop weak search", RED_LIGHT, RED),
    ]
    router_center = diagnosis[-1][0] + 0.062
    branch_y = 0.579
    arrow(ax, router_center, 0.663, router_center, branch_y, TEAL, 1.25)
    rule(ax, actions[0][0] + 0.072, branch_y, actions[-1][0] + 0.072, branch_y, TEAL, 1.0)
    for x, title_value, body, face, accent in actions:
        arrow(ax, x + 0.072, branch_y, x + 0.072, 0.550, TEAL, 0.95)
        node(x, 0.492, 0.144, 0.060, title_value, body, face, face, accent, 6.3, 4.7)

    # Action bus merges into deterministic execution.
    bus_y = 0.477
    for x, *_ in actions:
        rule(ax, x + 0.072, 0.492, x + 0.072, bus_y, "#9FB1B8", 0.8)
    rule(ax, actions[0][0] + 0.072, bus_y, actions[-1][0] + 0.072, bus_y, "#9FB1B8", 0.9)
    arrow(ax, 0.147, bus_y, 0.147, 0.446, NAVY_2, 1.0)

    execution = [
        (0.075, "Axis-isolated executor", "One research axis per matched experiment", 0.170),
        (0.282, "Validation evaluator", "Multi-seed · OOF · scaffold / OOD", 0.170),
        (0.489, "Evidence ledger", "Success + failure · cost · predictions · hashes", 0.170),
        (0.696, "Evidence decision", "Supported? · refuted? · inconclusive?", 0.170),
    ]
    for x, title_value, body, w in execution:
        node(x, 0.378, w, 0.068, title_value, body, WHITE, GRID, NAVY, 6.8, 4.9)
    for i in range(len(execution) - 1):
        arrow(ax, execution[i][0] + execution[i][3] + 0.004, 0.412,
              execution[i + 1][0] - 0.004, 0.412, NAVY_2, 1.05)

    # Explicit evidence feedback: negative / insufficient evidence returns to diagnosis or routing.
    text(ax, 0.779, 0.460, "REFUTED", 5.0, RED, "bold", ha="center")
    arrow(ax, 0.776, 0.448, 0.704, 0.661, RED, 0.95, rad=-0.30)
    text(ax, 0.861, 0.457, "INCONCLUSIVE", 5.0, GOLD, "bold", ha="center")
    arrow(ax, 0.837, 0.447, router_center, 0.661, GOLD, 0.95, rad=0.22)

    # Supported evidence crosses into the immutable certification plane.
    text(ax, 0.897, 0.399, "SUPPORTED", 5.2, GREEN, "bold", ha="center")
    arrow(ax, 0.870, 0.398, 0.870, 0.294, GREEN, 1.25)

    cert = [
        (0.075, "Candidate promotion", "Mean score · stability\nrisk · cost"),
        (0.247, "SHA-256 freeze lock", "Serialize the complete\nrecipe and source state"),
        (0.419, "Held-out certification", "Retrain from scratch\n≥5 seeds · read-only test"),
        (0.591, "Claim evaluator", "Prediction · transfer\nAgent increment · diagnosis"),
        (0.763, "Audit & evidence pack", "Scores · predictions · provenance\nreviewer bundle"),
    ]
    for x, title_value, body in cert:
        node(x, 0.190, 0.145, 0.074, title_value, body, NAVY, NAVY, WHITE, 6.4, 4.8)
    for i in range(len(cert) - 1):
        arrow(ax, cert[i][0] + 0.148, 0.227, cert[i + 1][0] - 0.004, 0.227, GOLD, 1.05)
    # Route the supported decision to promotion without crossing the sequence.
    rule(ax, 0.870, 0.294, 0.147, 0.294, GREEN, 1.0)
    arrow(ax, 0.147, 0.294, 0.147, 0.266, GREEN, 1.05)

    # One-shot firewall and future-audit branch.
    rule(ax, 0.055, 0.166, 0.943, 0.166, RED, 0.9, "--")
    text(ax, 0.057, 0.151, "ONE-SHOT TEST FIREWALL", 5.7, RED, "bold", va="top")
    text(ax, 0.943, 0.151, "official outcomes cannot mutate the same frozen lineage", 5.7, RED, "bold", ha="right", va="top")
    box(ax, 0.390, 0.098, 0.220, 0.041, WHITE, GOLD, radius=0.007, lw=0.8)
    text(ax, 0.500, 0.119, "New version → new future audit set", 6.2, "#805B1A", "bold", ha="center")
    arrow(ax, 0.835, 0.190, 0.610, 0.120, GOLD, 0.9, style="-|>", rad=0.16)

    footer(
        ax,
        "Adapted from docs/admet_evo_research_agent_architecture.md · implementation: runner.py / research_loop.py / freeze_audit.py",
        "Discovery uses train + validation only",
    )
    fig.savefig(out, dpi=180, facecolor=WHITE)
    plt.close(fig)

def short_task(name: str) -> str:
    mapping = {
        "Bioavailability_Ma": "Bioavailability",
        "CYP2C9_Substrate_CarbonMangels": "CYP2C9 substrate",
        "CYP2D6_Substrate_CarbonMangels": "CYP2D6 substrate",
        "CYP3A4_Substrate_CarbonMangels": "CYP3A4 substrate",
        "Clearance_Hepatocyte_AZ": "Hepatocyte clearance",
        "Clearance_Microsome_AZ": "Microsome clearance",
        "Lipophilicity_AstraZeneca": "Lipophilicity",
        "Pgp_Broccatelli": "P-gp",
        "Solubility_AqSolDB": "Solubility",
        "Half_Life_Obach": "Half-life",
        "VDss_Lombardo": "VDss",
        "Caco2_Wang": "Caco-2",
        "LD50_Zhu": "LD50",
    }
    return mapping.get(name, name.replace("_", " "))


def slide_performance(out: Path, scorecard: dict):
    fig, ax = canvas()
    status = scorecard["status"]
    header(
        ax,
        "Frozen performance | 22 endpoints",
        "Competitive coverage across the full TDC ADMET benchmark",
        "Five-seed frozen means inserted into the dated public leaderboard; lower is better for MAE, higher otherwise.",
        2,
    )

    cards = [
        ("TOP-5", f"{status['top5']} / 22", TEAL_DARK, "82% of endpoints"),
        ("TOP-10", f"{status['top10']} / 22", NAVY, "95% of endpoints"),
        ("MEAN RANK", f"{scorecard['rank_statistics']['mean_rank']:.2f}", GOLD, "across 22 tasks"),
        ("MEDIAN RANK", f"{scorecard['rank_statistics']['median_rank']:.0f}", NAVY_2, "public snapshot"),
    ]
    for idx, (label, value, accent, detail) in enumerate(cards):
        x = 0.055 + idx * 0.162
        box(ax, x, 0.694, 0.146, 0.092, WHITE, GRID, radius=0.009, lw=0.8)
        ax.add_patch(Rectangle((x, 0.694), 0.008, 0.092, color=accent, zorder=4))
        text(ax, x + 0.019, 0.760, label, 6.8, MUTED, "bold")
        text(ax, x + 0.019, 0.730, value, 16.5, accent, "bold")
        text(ax, x + 0.019, 0.705, detail, 6.2, MUTED)

    box(ax, 0.720, 0.694, 0.223, 0.092, NAVY, NAVY, radius=0.009, lw=0)
    text(ax, 0.737, 0.760, "RANK-1 ENDPOINTS", 6.8, "#C8DCE8", "bold")
    text(ax, 0.737, 0.731, "5", 16.5, WHITE, "bold")
    text(ax, 0.774, 0.731, "AMES · Half-life · Lipophilicity\nPPBR · Solubility", 6.5, WHITE, "bold", va="center")

    # Rank lollipop chart.
    tasks = sorted(scorecard["tasks"], key=lambda row: (row["estimated_rank"], short_task(row["task"])))
    chart_left, chart_right = 0.185, 0.705
    chart_bottom, chart_top = 0.106, 0.650
    n = len(tasks)
    ys = [chart_top - (i + 0.5) * (chart_top - chart_bottom) / n for i in range(n)]
    xrank = lambda rank: chart_left + (rank - 1) / 21 * (chart_right - chart_left)

    ax.add_patch(Rectangle((xrank(1), chart_bottom), xrank(5.5) - xrank(1), chart_top - chart_bottom,
                           color=GREEN_LIGHT, alpha=0.75, zorder=0))
    ax.add_patch(Rectangle((xrank(5.5), chart_bottom), xrank(10.5) - xrank(5.5), chart_top - chart_bottom,
                           color=GOLD_LIGHT, alpha=0.75, zorder=0))
    ax.add_patch(Rectangle((xrank(10.5), chart_bottom), xrank(22) - xrank(10.5), chart_top - chart_bottom,
                           color=RED_LIGHT, alpha=0.45, zorder=0))
    for rank in [1, 5, 10, 15, 20]:
        x = xrank(rank)
        rule(ax, x, chart_bottom, x, chart_top, WHITE if rank in (5, 10) else GRID, 0.8)
        text(ax, x, chart_top + 0.017, str(rank), 6.3, MUTED, ha="center")
    text(ax, chart_left, chart_top + 0.040, "ESTIMATED RANK ON TDC PUBLIC SNAPSHOT", 6.8, NAVY, "bold", va="bottom")
    text(ax, xrank(3), chart_bottom - 0.020, "TOP-5", 6.1, GREEN, "bold", ha="center")
    text(ax, xrank(8), chart_bottom - 0.020, "TOP-10", 6.1, "#8A641F", "bold", ha="center")

    for row, y in zip(tasks, ys):
        rank = row["estimated_rank"]
        color = TEAL_DARK if rank <= 5 else GOLD if rank <= 10 else RED
        text(ax, chart_left - 0.012, y, short_task(row["task"]), 5.7, INK,
             "bold" if rank == 1 else "normal", ha="right")
        rule(ax, chart_left, y, xrank(rank), y, "#B8C4C8", 0.65)
        ax.scatter([xrank(rank)], [y], s=26 if rank <= 5 else 22, color=color,
                   edgecolor=WHITE, linewidth=0.6, zorder=7)
        text(ax, xrank(rank) + 0.007, y, str(rank), 5.5, color, "bold", ha="left")

    # Interpretive evidence panel.
    box(ax, 0.746, 0.106, 0.197, 0.544, PALE, GRID, radius=0.010, lw=0.8)
    text(ax, 0.764, 0.621, "EVIDENCE OF STRENGTH", 7.2, NAVY, "bold", va="top")
    rule(ax, 0.764, 0.598, 0.925, 0.598, GRID, 0.7)

    callouts = [
        ("01", "Broad coverage", "18 Top-5 results across classification, regression and ranking endpoints.", TEAL_DARK),
        ("02", "Hard-task rescue", "CYP2C9 substrate: 0.340 → 0.434 ± 0.009 AUPRC; estimated rank 5.", NAVY_2),
        ("03", "Endpoint adaptation", "Rank-1 recipes span 2D ensembles, Uni-Mol2 and heterogeneous rank fusion.", GOLD),
    ]
    yy = 0.561
    for number, title_value, body, accent in callouts:
        box(ax, 0.764, yy - 0.015, 0.026, 0.026, accent, accent, radius=0.005, lw=0)
        text(ax, 0.777, yy - 0.002, number, 5.7, WHITE, "bold", ha="center")
        text(ax, 0.800, yy + 0.004, title_value, 7.3, accent, "bold")
        text(ax, 0.800, yy - 0.029, body, 6.5, INK, va="top", linespacing=1.23)
        yy -= 0.115

    box(ax, 0.764, 0.166, 0.161, 0.128, RED_LIGHT, RED_LIGHT, radius=0.008, lw=0)
    text(ax, 0.776, 0.271, "CONSERVATIVE EXCEPTION", 6.4, RED, "bold", va="top")
    text(ax, 0.776, 0.238, "Bioavailability", 8.3, RED, "bold", va="top")
    text(ax, 0.776, 0.205, "0.731 ± 0.007 AUROC · rank 21", 6.4, INK, "bold", va="top")
    text(ax, 0.776, 0.179, "Displayed status retained pending an\nauthoritative TDC page correction.", 5.8, MUTED, va="top")

    footer(
        ax,
        "Source: reports/goal_scorecard_v27.json; TDC public leaderboard snapshot captured 17 Aug 2026",
        "Estimated insertion ranks; not an official TDC submission",
    )
    fig.savefig(out, dpi=180, facecolor=WHITE)
    plt.close(fig)


def fmt_ci(values):
    return f"[{values[0]:+.5f}, {values[1]:+.5f}]"


def slide_autonomy(out: Path, router: dict, guard: dict, faults: dict, policy: dict):
    fig, ax = canvas()
    header(
        ax,
        "Autonomous research | Evidence table",
        "Autonomy evidence: strong protocol execution and cost control",
        "All comparisons preserve the validation/test firewall; failed preregistered gates are retained as negative evidence.",
        3,
    )

    mem = router["memory_ablation"]
    rnd = router["matched_cost_random"]
    grd = guard["primary_guard_minus_unguarded"]
    fault = faults["score"]
    strong = policy["comparisons"]["strong_fixed"]

    # Table geometry.
    left, right = 0.055, 0.945
    top, bottom = 0.768, 0.230
    widths = [0.205, 0.160, 0.235, 0.165, 0.125]
    xs = [left]
    for w in widths:
        xs.append(xs[-1] + w)
    row_h = (top - bottom) / 6

    ax.add_patch(Rectangle((left, top - row_h), right - left, row_h, color=NAVY, zorder=1))
    headers = ["Capability / control", "Protocol", "Observed effect", "Efficiency / reliability", "Evidence status"]
    for i, label in enumerate(headers):
        text(ax, xs[i] + 0.010, top - row_h / 2, label, 7.0, WHITE, "bold")

    rows = [
        (
            "Scientific fault detection",
            "10 deterministic\nfault-injection cases",
            f"Micro F1  {fault['micro_f1']:.3f}\nCritical recall  {fault['critical_recall']:.3f}",
            f"Mean detection  {fault['mean_detection_seconds']:.2f} s\nValid remediation  {fault['remediation_validity']:.3f}",
            "PASSED*", GREEN, GREEN_LIGHT,
        ),
        (
            "Full-outcome memory\nvs winner-only memory",
            "22 tasks\nfresh validation",
            f"Mean Δ  {mem['summary']['mean']:+.5f}\n95% CI  {fmt_ci(mem['summary']['bootstrap_mean_95_ci'])}\nW/T/L  4 / 12 / 6",
            f"Selected compute  {100*mem['aggregate_full_to_winner_cost_ratio']:.1f}%\nof winner-only memory",
            "COST SUPPORTED\nACCURACY GATE FAILED", GOLD, GOLD_LIGHT,
        ),
        (
            "Agent router\nvs matched-cost random",
            f"22 tasks · {rnd['heldout_decisions']} held-out\ndecisions",
            f"Mean Δ  {rnd['summary']['mean']:+.5f}\n95% CI  {fmt_ci(rnd['summary']['bootstrap_mean_95_ci'])}\nW/T/L  7 / 5 / 10 · p={rnd['summary']['wilcoxon_p_value']:.3f}",
            "Expected training cost\nmatched to numerical precision",
            "SUPERIORITY\nNOT SUPPORTED", RED, RED_LIGHT,
        ),
        (
            "Metric-tail guard\nvs unguarded router",
            "22 tasks\nfresh seeds 4–5",
            f"Mean Δ  {grd['mean']:+.5f}\n95% CI  {fmt_ci(grd['bootstrap_mean_95_ci'])}\nW/T/L  0 / 22 / 0",
            f"Fit-time ratio  {100*guard['claim_gate']['aggregate_guard_cost_ratio_vs_score']:.1f}%\nvs score selection",
            "SAFETY GAIN\nNOT SUPPORTED", RED, RED_LIGHT,
        ),
        (
            "Agent search\nvs strong fixed model",
            "22 tasks\nunmatched action budget",
            f"Mean Δ  {strong['mean_oriented_delta']:+.4f}\n95% CI  {fmt_ci(strong['bootstrap_95_ci'])}\nW/T/L  19 / 3 / 0",
            f"Compute ratio  {strong['mean_compute_ratio']:.1f}×\nAction ratio  {strong['mean_action_ratio']:.1f}×",
            "PROMISING SIGNAL\nNOT A FAIR CLAIM", NAVY_2, TEAL_LIGHT,
        ),
    ]

    for r, row in enumerate(rows):
        y0 = top - (r + 2) * row_h
        fill = WHITE if r % 2 == 0 else PALE
        ax.add_patch(Rectangle((left, y0), right - left, row_h, color=fill, zorder=0))
        for x in xs:
            rule(ax, x, y0, x, y0 + row_h, GRID, 0.55)
        rule(ax, left, y0, right, y0, GRID, 0.55)
        capability, protocol_text, effect_text, efficiency_text, status_text, accent, status_fill = row
        text(ax, xs[0] + 0.010, y0 + row_h / 2, capability, 7.0, NAVY, "bold", va="center")
        text(ax, xs[1] + 0.010, y0 + row_h / 2, protocol_text, 6.4, MUTED, va="center")
        text(ax, xs[2] + 0.010, y0 + row_h / 2, effect_text, 6.3, INK, va="center", linespacing=1.25)
        text(ax, xs[3] + 0.010, y0 + row_h / 2, efficiency_text, 6.3, INK, va="center", linespacing=1.25)
        box(ax, xs[4] + 0.010, y0 + 0.020, widths[4] - 0.020, row_h - 0.040,
            status_fill, status_fill, radius=0.006, lw=0)
        text(ax, xs[4] + widths[4] / 2, y0 + row_h / 2, status_text, 5.8, accent, "bold", ha="center", linespacing=1.2)
    rule(ax, right, bottom, right, top, GRID, 0.55)
    rule(ax, left, top, right, top, GRID, 0.55)

    # Conclusion strip.
    box(ax, 0.055, 0.078, 0.890, 0.112, NAVY, NAVY, radius=0.010, lw=0)
    text(ax, 0.075, 0.166, "EVIDENCE-GRADE CONCLUSION", 7.0, "#BFD9E4", "bold", va="top")
    text(ax, 0.075, 0.128,
         "SUPPORTED  |  protocol safety · auditable experimentation · 22-task diagnosis · strong frozen coverage · aggregate cost control",
         7.2, WHITE, "bold")
    text(ax, 0.075, 0.097,
         "NOT YET SUPPORTED  |  superior action selection at matched cost · independent blind autonomous-science performance",
         7.2, "#F0C7C3", "bold")
    text(ax, 0.932, 0.166, "*Internal fixture, not an independent blind challenge", 5.6, "#BFD9E4", ha="right", va="top")

    footer(
        ax,
        "Sources: runs/v27-router-controls/report.json; v27-router-risk-guard/report.json; reports/v3_hidden_fault_suite.json",
        "Oriented Δ: positive favors ADMET-EvO",
    )
    fig.savefig(out, dpi=180, facecolor=WHITE)
    plt.close(fig)


def build_ppt(images: list[Path], output: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "ADMET-EvO: Architecture, TDC-22 Performance, and Autonomous Research Evidence"
    prs.core_properties.subject = "Four-slide scientific overview of ADMET-EvO v2.7/v3"
    prs.core_properties.author = "ADMET-EvO Research Team"
    prs.core_properties.keywords = "ADMET, TDC, autonomous research agent, molecular property prediction"
    blank = prs.slide_layouts[6]
    for path in images:
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(str(path), 0, 0, width=prs.slide_width, height=prs.slide_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    scorecard = load_json(SCORECARD)
    router = load_json(ROUTER_CONTROLS)
    guard = load_json(RISK_GUARD)
    faults = load_json(FAULT_SUITE)
    policy = load_json(POLICY_COMPARISON)
    images = [ASSETS / f"slide_{index}.png" for index in range(1, 5)]
    slide_architecture(images[0])
    slide_performance(images[1], scorecard)
    slide_autonomy(images[2], router, guard, faults, policy)
    slide_architecture_full(images[3])
    build_ppt(images, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "slides": len(images),
        "assets": [str(path) for path in images],
        "snapshot": scorecard["leaderboard_captured_at"],
        "freeze_sha256": scorecard["freeze_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
