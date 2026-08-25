#!/usr/bin/env python3
"""Build publication-grade Methods figures for the ADMET-EvO paper."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "methods" / "figures"

NAVY = "#204F73"
TEAL = "#16827D"
BLUE = "#4C78A8"
GOLD = "#D49A24"
RED = "#C4514A"
PURPLE = "#7560A8"
DARK = "#17212B"
GRAY = "#66717C"
MID = "#AAB3BC"
LIGHT = "#F4F7F9"
WHITE = "#FFFFFF"


def configure():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.7,
        "axes.edgecolor": MID, "figure.facecolor": WHITE,
        "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
        "svg.hashsalt": "admet-evo-methods-v1",
    })


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def box(ax, x, y, w, h, title, subtitle="", color=NAVY, fontsize=7.1):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=0.9, edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.09), zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*(0.62 if subtitle else 0.5), title,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=DARK, linespacing=0.92, zorder=3)
    if subtitle:
        ax.text(x+w/2, y+h*0.23, subtitle, ha="center", va="center",
                fontsize=5.9, color=GRAY, linespacing=0.9, zorder=3)


def arrow(ax, p1, p2, color=GRAY, lw=0.9, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=9, linewidth=lw,
        color=color, shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
    ))


def panel(ax, label, title):
    ax.text(-0.015, 1.025, label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.0, 1.025, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=DARK)


def figure1_architecture():
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.0], hspace=0.27)

    ax = fig.add_subplot(gs[0])
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    panel(ax, "a", "System architecture: a protocol-constrained autonomous scientist")

    nodes = [
        (0.015, "Task intake", "structure · label", NAVY),
        (0.175, "Endpoint contract", "semantics · split", NAVY),
        (0.345, "Research state", "hardness · faults", TEAL),
        (0.515, "Hypothesis engine", "effect · cost", TEAL),
        (0.685, "Action router", "one axis", TEAL),
    ]
    for x,t,s,c in nodes:
        box(ax,x,0.67,0.14,0.18,t,s,c)
    for i in range(len(nodes)-1):
        arrow(ax,(nodes[i][0]+0.14,0.76),(nodes[i+1][0],0.76))

    axes = [
        ("Feature","repr.",BLUE), ("Model","learner / loss",PURPLE),
        ("Data","search / clean",TEAL),
        ("Replicate","seed / OOD",GOLD), ("Stop","freeze / rollback",RED),
    ]
    xs = [0.02,0.17,0.32,0.47,0.62]
    for x,(t,s,c) in zip(xs,axes):
        box(ax,x,0.33,0.13,0.16,t,s,c,6.7)
        arrow(ax,(0.755,0.67),(x+0.065,0.49),MID,0.7)
    box(ax,0.77,0.33,0.095,0.16,"Matched\nexecutor","control vs treatment",NAVY,6.5)
    box(ax,0.885,0.33,0.10,0.16,"Evidence","effect · cost",GOLD,6.6)
    arrow(ax,(0.75,0.41),(0.77,0.41)); arrow(ax,(0.865,0.41),(0.885,0.41))
    box(ax,0.77,0.06,0.095,0.15,"Ledger","hash · lineage",GOLD,6.7)
    box(ax,0.885,0.06,0.10,0.15,"Memory","update · reject",GOLD,6.7)
    arrow(ax,(0.935,0.33),(0.82,0.21),GOLD)
    arrow(ax,(0.865,0.135),(0.885,0.135),GOLD)
    arrow(ax,(0.935,0.21),(0.59,0.67),GOLD,0.9,0.22)
    ax.text(0.02,0.13,
            "Toolkit: fingerprints + descriptors; tree, boosting and kernel learners",
            fontsize=6.4,color=GRAY)
    ax.text(0.02,0.06,
            "The optimized object is a research policy; endpoint-specific predictors are frozen products.",
            fontsize=6.8,color=DARK,fontweight="bold")

    ax = fig.add_subplot(gs[1])
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    panel(ax, "b", "Task-conditioned data search, cleaning and dataset expansion")

    row1 = [
        (0.01,"Endpoint\ncontract","assay · unit",NAVY),
        (0.16,"Source scout","DB · papers",TEAL),
        (0.31,"Query ledger","query · license",TEAL),
        (0.46,"Immutable raw","record · hash",TEAL),
        (0.61,"Semantic gates","endpoint · threshold",GOLD),
        (0.79,"Compatibility","direct · aux · reject",GOLD),
    ]
    widths=[0.12,0.12,0.12,0.12,0.15,0.19]
    for (x,t,s,c),w in zip(row1,widths): box(ax,x,0.69,w,0.17,t,s,c,6.6)
    for i in range(len(row1)-1):
        arrow(ax,(row1[i][0]+widths[i],0.775),(row1[i+1][0],0.775),
              TEAL if i<3 else GOLD)

    row2 = [
        (0.01,"Standardize","salt · identity",BLUE),
        (0.19,"Resolve conflicts","replicate · lineage",BLUE),
        (0.38,"Structure firewall","exact · similar",RED),
        (0.59,"Domain audit","scaffold · shift",PURPLE),
        (0.79,"Modeling role","pool · quarantine",GOLD),
    ]
    widths2=[0.15,0.16,0.18,0.17,0.19]
    for (x,t,s,c),w in zip(row2,widths2): box(ax,x,0.36,w,0.17,t,s,c,6.5)
    for i in range(len(row2)-1):
        arrow(ax,(row2[i][0]+widths2[i],0.445),(row2[i+1][0],0.445),MID)
    arrow(ax,(0.885,0.69),(0.885,0.53),GOLD)

    box(ax,0.15,0.06,0.24,0.16,"Matched augmentation test",
        "fixed recipe · baseline vs expanded",TEAL,6.8)
    box(ax,0.46,0.06,0.18,0.16,"Evidence status",
        "supported · refuted · unclear",GOLD,6.8)
    box(ax,0.71,0.06,0.22,0.16,"Versioned evidence bundle",
        "raw · exclusions · audit · manifest",NAVY,6.7)
    arrow(ax,(0.885,0.36),(0.39,0.14),TEAL,0.9,0.08)
    arrow(ax,(0.39,0.14),(0.46,0.14),TEAL)
    arrow(ax,(0.64,0.14),(0.71,0.14),GOLD)
    ax.text(0.015,0.59,"Endpoint-name agreement alone never authorizes merging.",
            fontsize=6.0,color=RED,fontweight="bold")
    ax.text(0.53,0.59,
            "Scaling pilots: Morgan radius 2, 2,048 bits; max Tanimoto < 0.80.",
            fontsize=5.7,color=GRAY)

    ax = fig.add_subplot(gs[2])
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    panel(ax, "c", "Evidence-gated research episode and immutable certification")

    centers=[0.07,0.22,0.37,0.52]
    labels=[("Diagnose","multidimensional state"),("Hypothesize","falsification test"),
            ("Execute","matched seeds"),("Evaluate","oriented effect")]
    for x,(t,s) in zip(centers,labels):
        box(ax,x-0.055,0.65,0.11,0.17,t,s,NAVY if x<0.3 else TEAL,6.5)
    for a,b in zip(centers[:-1],centers[1:]):
        arrow(ax,(a+0.055,0.735),(b-0.055,0.735))

    circ=Circle((0.66,0.735),0.065,facecolor=mpl.colors.to_rgba(GOLD,0.12),
                edgecolor=GOLD,lw=1.0)
    ax.add_patch(circ)
    ax.text(0.66,0.735,"Evidence\ngate",ha="center",va="center",
            fontsize=6.7,fontweight="bold",color=DARK)
    arrow(ax,(0.575,0.735),(0.595,0.735),GOLD)

    outcomes=[
        (0.76,0.76,"Supported","promote + remember",TEAL),
        (0.76,0.53,"Refuted","correct + rollback",RED),
        (0.76,0.30,"Inconclusive","replicate or stop",GOLD),
    ]
    for x,y,t,s,c in outcomes:
        box(ax,x,y,0.20,0.14,t,s,c,6.7)
    arrow(ax,(0.715,0.77),(0.76,0.83),TEAL)
    arrow(ax,(0.72,0.72),(0.76,0.60),RED)
    arrow(ax,(0.70,0.69),(0.76,0.37),GOLD)
    arrow(ax,(0.76,0.60),(0.22,0.65),RED,0.9,-0.18)
    arrow(ax,(0.76,0.37),(0.22,0.65),GOLD,0.9,-0.30)

    box(ax,0.08,0.12,0.17,0.15,"Candidate freeze","recipe · code · SHA-256",RED,6.7)
    box(ax,0.34,0.12,0.17,0.15,"Blind certification","read-only evaluation",RED,6.7)
    box(ax,0.60,0.12,0.17,0.15,"Claim package","effect · cost · failure",NAVY,6.7)
    arrow(ax,(0.86,0.76),(0.165,0.27),TEAL,0.9,0.24)
    arrow(ax,(0.25,0.195),(0.34,0.195),RED)
    arrow(ax,(0.51,0.195),(0.60,0.195),RED)
    ax.plot([0.295,0.295],[0.07,0.32],color=RED,lw=1.1,ls=(0,(4,3)))
    ax.text(0.30,0.075,"no feedback to discovery",fontsize=6.2,color=RED)

    fig.suptitle("ADMET-EvO autonomous research architecture",x=0.06,y=0.995,
                 ha="left",fontsize=12,fontweight="bold",color=DARK)
    save(fig,"figure1_autonomous_architecture_and_workflow")


def load_json(name):
    return json.loads((ROOT/name).read_text())


def figure2_axis_feedback():
    grid=load_json("reports/transport_grid_results_v1.json")
    confirm=load_json("reports/post_memory_confirmation_v1.json")
    efficiency=load_json("reports/practical_efficiency_certification_v1.json")

    fig=plt.figure(figsize=(7.2,7.1))
    gs=fig.add_gridspec(2,2,hspace=0.38,wspace=0.40)

    # A: action effect matrix
    ax=fig.add_subplot(gs[0,0])
    panel(ax,"a","Axis-isolated action effects")
    tasks=["Caco2_Wang","BBB_Martins","CYP2C9_Veith",
           "Clearance_Hepatocyte_AZ","LD50_Zhu"]
    short=["Caco-2","BBB","CYP2C9","Hepatic CL","LD50"]
    acts=["feature-domain-shift","model-estimator-family","data-quality-filter"]
    act_labels=["Feature","Model","Data filter"]
    lookup={(e["task"],e["action_id"]):e for e in grid["episodes"]}
    values=np.array([[lookup[(t,a)]["mean_oriented_delta"] for a in acts] for t in tasks])
    status=[[lookup[(t,a)]["status"] for a in acts] for t in tasks]
    cmap=LinearSegmentedColormap.from_list("effect",[RED,WHITE,TEAL])
    norm=TwoSlopeNorm(vmin=-0.045,vcenter=0,vmax=0.03)
    im=ax.imshow(values,cmap=cmap,norm=norm,aspect="auto")
    ax.set_xticks(range(3),act_labels); ax.set_yticks(range(5),short)
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(3):
            symbol={"supported":"S","refuted":"R","inconclusive":"I"}[status[i][j]]
            color=WHITE if abs(values[i,j])>0.018 else DARK
            ax.text(j,i,f"{values[i,j]:+.3f}\n{symbol}",ha="center",va="center",
                    fontsize=6.6,color=color,fontweight="bold")
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03)
    cb.set_label("Oriented validation Δ",fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.text(0,-0.19,"S supported   R refuted   I inconclusive",
            transform=ax.transAxes,fontsize=5.7,color=GRAY)

    # B: task-conditioned data scaling
    ax=fig.add_subplot(gs[0,1])
    panel(ax,"b","Task-conditioned data expansion")
    data=[
        ("hERG",0.10642010345415849,43669),
        ("DILI",0.0633218800531462,618),
        ("Caco-2",0.026070980318822767,None),
        ("BBB",0.007120831740322815,4367),
    ]
    names=[x[0] for x in data][::-1]
    vals=np.array([x[1] for x in data][::-1])
    rows=[x[2] for x in data][::-1]
    y=np.arange(len(names))
    ax.barh(y,vals,color=TEAL,height=0.55)
    ax.axvline(0,color=DARK,lw=0.8)
    ax.axvline(0.005,color=GOLD,lw=0.8,ls="--")
    ax.set_yticks(y,names); ax.set_xlabel("Oriented validation Δ after expansion")
    ax.set_xlim(0,0.12); ax.grid(axis="x",color=LIGHT,lw=0.8)
    ax.spines[["top","right","left"]].set_visible(False)
    for yi,v,n in zip(y,vals,rows):
        suffix=f"  n={n:,}" if n is not None else ""
        ax.text(v+0.002,yi,f"{v:+.3f}{suffix}",va="center",fontsize=6.6,color=DARK)
    ax.text(0,-0.19,"Fixed recipes; exact/near-neighbour firewall; discovery validation only",
            transform=ax.transAxes,fontsize=6.0,color=GRAY)

    # C: independent confirmation and memory rollback
    ax=fig.add_subplot(gs[1,0])
    panel(ax,"c","Confirmation revises action memory")
    eps=confirm["episodes"]
    order=sorted(eps,key=lambda x:x["mean_oriented_delta"])
    names=[e["task"].replace("_AqSolDB","").replace("_AZ","").replace("_Veith","")
           .replace("_Microsome"," micro") for e in order]
    vals=np.array([e["mean_oriented_delta"] for e in order])
    st=[e["status"] for e in order]
    colors=[{"supported":TEAL,"refuted":RED,"inconclusive":GOLD}[s] for s in st]
    y=np.arange(len(order))
    ax.axvspan(-0.005,0.005,color=mpl.colors.to_rgba(GOLD,0.10),zorder=0)
    ax.axvline(0,color=DARK,lw=0.8)
    ax.hlines(y,0,vals,color=[mpl.colors.to_rgba(c,0.55) for c in colors],lw=1.6)
    ax.scatter(vals,y,c=colors,s=34,edgecolor=WHITE,lw=0.6,zorder=3)
    ax.set_yticks(y,names); ax.set_xlabel("Confirmed oriented Δ")
    ax.set_xlim(-0.04,0.285); ax.grid(axis="x",color=LIGHT,lw=0.8)
    ax.spines[["top","right","left"]].set_visible(False)
    for yi,v,s in zip(y,vals,st):
        ax.text(v+0.006,yi,f"{v:+.3f}",
                ha="left",va="center",fontsize=6.4,color=DARK)
    ax.text(0,-0.20,"Max pruning regret = 0.259 → global negative-memory pruning disabled",
            transform=ax.transAxes,fontsize=6.2,color=RED,fontweight="bold")

    # D: stop/cost feedback
    ax=fig.add_subplot(gs[1,1])
    panel(ax,"d","Stop policy reduces compute")
    ax.set_xlim(-0.006,0.006); ax.set_ylim(-0.2,1.6)
    ax.axvspan(-0.005,0.005,color=mpl.colors.to_rgba(TEAL,0.08))
    ax.axvline(0,color=DARK,lw=0.8)
    mean=efficiency["mean_oriented_gain"]
    lo,hi=efficiency["bootstrap_95_ci"]
    ax.errorbar(mean,1.05,xerr=[[mean-lo],[hi-mean]],fmt="o",color=NAVY,
                ecolor=NAVY,capsize=3,ms=5)
    ax.text(mean,1.27,f"Δ={mean:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]",
            ha="center",va="bottom",fontsize=6.5,color=DARK)
    ax.set_yticks([1.05],["Performance"])
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",color=LIGHT,lw=0.8)
    inset=ax.inset_axes([0.05,0.02,0.90,0.30])
    inset.barh([0], [1.0], color=MID, height=0.42)
    inset.barh([0], [efficiency["aggregate_cost_ratio"]], color=TEAL, height=0.42)
    inset.set_xlim(0,1.05); inset.set_yticks([])
    inset.set_xlabel("Relative selected-recipe compute",fontsize=6.2,labelpad=1)
    inset.tick_params(axis="x",labelsize=6)
    inset.spines[["top","right","left"]].set_visible(False)
    inset.text(1.0,0.18,"Score-only 1.00×",ha="right",va="bottom",fontsize=6.0,color=GRAY)
    inset.text(efficiency["aggregate_cost_ratio"],-0.18,
               f"Agent {efficiency['aggregate_cost_ratio']:.3f}×  (−{efficiency['estimated_cost_reduction']*100:.1f}%)",
               ha="left",va="top",fontsize=6.2,color=TEAL,fontweight="bold")

    fig.suptitle("Research-action effects are measured and returned to the policy",
                 x=0.06,y=0.995,ha="left",fontsize=12,fontweight="bold",color=DARK)
    save(fig,"figure2_multi_axis_optimization_feedback")


if __name__=="__main__":
    configure()
    figure1_architecture()
    figure2_axis_feedback()
    print(OUT)
