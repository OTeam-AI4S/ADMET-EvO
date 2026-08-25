#!/usr/bin/env python3
"""Nature-style vector figures for the ADMET-EvO Methods section."""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "methods" / "figures"

INK = "#202124"
GREY = "#6B7075"
LIGHT_GREY = "#D9DDE1"
PALE = "#F5F6F7"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#D55E00"
GOLD = "#E69F00"
PURPLE = "#8C6BB1"
WHITE = "#FFFFFF"


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.65,
        "axes.edgecolor": GREY,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "svg.hashsalt": "admet-evo-nature-v2",
    })


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        kw = {"bbox_inches": "tight", "pad_inches": 0.025}
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)


def panel(ax, letter, title):
    ax.text(-0.035, 1.03, letter, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", ha="left", va="bottom", color=INK)
    ax.text(0.045, 1.03, title, transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", ha="left", va="bottom", color=INK)


def arrow(ax, p1, p2, color=GREY, lw=0.85, rad=0.0, mutation=8):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, shrinkA=1.5, shrinkB=1.5,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
    ))


def pill(ax, x, y, w, h, text, edge=LIGHT_GREY, face=WHITE, color=INK, size=6.6):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=face,edgecolor=edge,lw=0.75,zorder=1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,color=color,zorder=2)


def figure1():
    fig = plt.figure(figsize=(7.2, 5.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.92], hspace=0.24)

    # a: research-state loop
    ax = fig.add_subplot(gs[0])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    panel(ax, "a", "ADMET-EvO optimizes a research policy around an explicit research state")

    cx, cy, radius = 0.47, 0.55, 0.245
    ax.add_patch(Circle((cx,cy),0.115,facecolor="#EEF5F8",edgecolor=BLUE,lw=1.1,zorder=3))
    ax.text(cx,cy+0.018,"Research state",ha="center",va="center",
            fontsize=8.3,fontweight="bold",color=INK,zorder=4)
    ax.text(cx,cy-0.035,"contract · hardness · faults\nmemory · budget · provenance",
            ha="center",va="center",fontsize=5.9,color=GREY,linespacing=1.0,zorder=4)

    stages = [
        ("Contract",130,BLUE), ("Diagnose",90,BLUE), ("Hypothesize",40,GREEN),
        ("Intervene",-40,GREEN), ("Evaluate",-90,GOLD), ("Revise",-130,ORANGE),
    ]
    points = []
    for label,deg,color in stages:
        th=math.radians(deg)
        x,y=cx+radius*math.cos(th),cy+radius*math.sin(th)
        points.append((x,y))
        ax.add_patch(Circle((x,y),0.031,facecolor=WHITE,edgecolor=color,lw=1.25,zorder=4))
        ax.text(x,y,str(len(points)),ha="center",va="center",fontsize=6.2,
                fontweight="bold",color=color,zorder=5)
        # labels placed radially
        lx,ly=cx+(radius+0.075)*math.cos(th),cy+(radius+0.075)*math.sin(th)
        ax.text(lx,ly,label,ha="center",va="center",fontsize=6.8,
                fontweight="bold",color=INK)
    for i in range(len(points)):
        arrow(ax,points[i],points[(i+1)%len(points)],color=GREY,lw=0.8,rad=-0.20)

    # inputs
    ax.text(0.03,0.79,"Scientific input",fontsize=6.1,color=GREY,fontweight="bold")
    pill(ax,0.03,0.65,0.15,0.10,"Target endpoint",edge=BLUE,face="#F3F8FB")
    pill(ax,0.03,0.52,0.15,0.10,"Internal data",edge=BLUE,face="#F3F8FB")
    pill(ax,0.03,0.39,0.15,0.10,"External evidence",edge=GREEN,face="#F1F8F5")
    arrow(ax,(0.18,0.70),(points[0][0]-0.03,points[0][1]),BLUE)
    arrow(ax,(0.18,0.57),(cx-0.12,cy),BLUE)
    arrow(ax,(0.18,0.44),(0.30,0.34),GREEN)

    # action spectrum
    ax.text(0.25,0.185,"Single-axis action",fontsize=6.1,color=GREY,fontweight="bold")
    actions=[("Feature",BLUE),("Model",PURPLE),("Data",GREEN),("Replicate",GOLD),("Stop",ORANGE)]
    x0=0.25
    for i,(name,color) in enumerate(actions):
        x=x0+i*0.095
        ax.plot([x,x+0.055],[0.135,0.135],lw=3.2,color=color,solid_capstyle="butt")
        ax.text(x+0.0275,0.10,name,ha="center",va="top",fontsize=6.1,color=INK)
    arrow(ax,(points[3][0],points[3][1]-0.03),(0.47,0.18),GREEN,rad=0.08)

    # outcomes and freeze
    ax.text(0.73,0.83,"Evidence state",fontsize=6.1,color=GREY,fontweight="bold")
    for j,(name,color) in enumerate([("supported",GREEN),("refuted",ORANGE),("inconclusive",GOLD)]):
        y=0.73-j*0.10
        ax.plot([0.73,0.75],[y,y],lw=3,color=color)
        ax.text(0.765,y,name,va="center",fontsize=6.4,color=INK)
    arrow(ax,(points[4][0]+0.02,points[4][1]),(0.73,0.63),GOLD,rad=-0.12)

    ax.plot([0.87,0.87],[0.25,0.82],color=ORANGE,lw=1.0,ls=(0,(3,2)))
    ax.text(0.875,0.80,"freeze",fontsize=6.0,color=ORANGE,rotation=90,va="top")
    pill(ax,0.895,0.57,0.09,0.12,"Frozen\npredictor",edge=ORANGE,face="#FCF4F1",size=6.3)
    pill(ax,0.895,0.38,0.09,0.12,"Blind\ncertificate",edge=ORANGE,face="#FCF4F1",size=6.3)
    arrow(ax,(0.84,0.63),(0.895,0.63),ORANGE)
    arrow(ax,(0.94,0.57),(0.94,0.50),ORANGE)
    ax.text(0.70,0.26,"Every transition is hash-linked;\nfailed and negative experiments are retained.",
            fontsize=6.1,color=GREY,ha="left")

    # b: external evidence track
    ax = fig.add_subplot(gs[1])
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    panel(ax, "b", "Task-conditioned data search, cleaning and expansion is an evidence-admission process")

    source_y=[0.73,0.55,0.37]
    source_names=["Databases","Primary literature","Regulatory sources"]
    source_colors=[GREEN,SKY,PURPLE]
    for y,name,color in zip(source_y,source_names,source_colors):
        ax.scatter([0.04],[y],s=28,color=color,zorder=3)
        ax.text(0.065,y,name,va="center",fontsize=6.5,color=INK)
        arrow(ax,(0.20,y),(0.25,0.55),color,0.7)

    xnodes=[0.27,0.40,0.53,0.66,0.79]
    nums=["1","2","3","4","5"]
    names=["Define","Preserve","Harmonize","Decontaminate","Assign role"]
    subs=["endpoint\nquery","raw\nprovenance","assay\nidentity","exact\nanalogue","direct · aux\nreject"]
    colors=[BLUE,GREEN,GOLD,ORANGE,PURPLE]
    for x,n,name,sub,color in zip(xnodes,nums,names,subs,colors):
        ax.add_patch(Circle((x,0.55),0.028,facecolor=WHITE,edgecolor=color,lw=1.2,zorder=3))
        ax.text(x,0.55,n,ha="center",va="center",fontsize=6.0,fontweight="bold",color=color)
        ax.text(x,0.72,name,ha="center",va="center",fontsize=6.5,fontweight="bold",color=INK)
        ax.text(x,0.37,sub,ha="center",va="center",fontsize=5.7,color=GREY)
    for a,b in zip(xnodes[:-1],xnodes[1:]):
        arrow(ax,(a+0.029,0.55),(b-0.029,0.55),GREY,0.8)

    arrow(ax,(0.82,0.55),(0.87,0.55),GREEN)
    ax.add_patch(Polygon([[0.87,0.43],[0.98,0.43],[0.98,0.67],[0.87,0.67]],
                         closed=True,facecolor="#F1F8F5",edgecolor=GREEN,lw=0.9))
    ax.text(0.925,0.575,"Matched data\nexperiment",ha="center",va="center",
            fontsize=6.6,fontweight="bold",color=INK)
    ax.text(0.925,0.475,"baseline ↔ expanded",ha="center",va="center",
            fontsize=5.7,color=GREY)

    ax.plot([0.53,0.98],[0.18,0.18],color=LIGHT_GREY,lw=0.8)
    ax.text(0.53,0.05,"Predeclared pilot firewall: Morgan radius 2, 2,048 bits; max Tanimoto < 0.80",
            fontsize=5.9,color=GREY,ha="left")
    ax.text(0.04,0.14,"Endpoint-name agreement alone never authorizes merging.",
            fontsize=6.1,color=ORANGE,fontweight="bold",ha="left")

    save(fig,"figure1_nature_autonomous_research_architecture")


def load(path):
    return json.loads((ROOT/path).read_text())


def figure2():
    grid=load("reports/transport_grid_results_v1.json")
    confirm=load("reports/post_memory_confirmation_v1.json")
    efficiency=load("reports/practical_efficiency_certification_v1.json")

    fig=plt.figure(figsize=(7.2,5.55))
    gs=fig.add_gridspec(2,2,hspace=0.52,wspace=0.44)

    # a
    ax=fig.add_subplot(gs[0,0]); panel(ax,"a","Axis-isolated intervention effects")
    tasks=["Caco2_Wang","BBB_Martins","CYP2C9_Veith","Clearance_Hepatocyte_AZ","LD50_Zhu"]
    short=["Caco-2","BBB","CYP2C9","Hepatic CL","LD50"]
    acts=["feature-domain-shift","model-estimator-family","data-quality-filter"]
    lookup={(e["task"],e["action_id"]):e for e in grid["episodes"]}
    z=np.array([[lookup[(t,a)]["mean_oriented_delta"] for a in acts] for t in tasks])
    st=[[lookup[(t,a)]["status"] for a in acts] for t in tasks]
    cmap=LinearSegmentedColormap.from_list("oi",[ORANGE,WHITE,GREEN])
    im=ax.imshow(z,cmap=cmap,norm=TwoSlopeNorm(vmin=-0.045,vcenter=0,vmax=0.03),aspect="auto")
    ax.set_xticks(range(3),["Feature","Model","Data filter"])
    ax.set_yticks(range(5),short); ax.tick_params(length=0)
    for i in range(5):
        for j in range(3):
            code={"supported":"S","refuted":"R","inconclusive":"I"}[st[i][j]]
            ax.text(j,i,f"{z[i,j]:+.3f}\n{code}",ha="center",va="center",
                    fontsize=6.0,fontweight="bold",
                    color=WHITE if abs(z[i,j])>0.022 else INK)
    cb=fig.colorbar(im,ax=ax,fraction=0.040,pad=0.025)
    cb.set_label("Oriented validation Δ",fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)
    ax.text(0,-0.15,"S supported   R refuted   I inconclusive",
            transform=ax.transAxes,fontsize=5.6,color=GREY)

    # b
    ax=fig.add_subplot(gs[0,1]); panel(ax,"b","Task-conditioned data expansion")
    data=[("hERG",0.10642010345415849,43669),("DILI",0.0633218800531462,618),
          ("Caco-2",0.026070980318822767,None),("BBB",0.007120831740322815,4367)]
    data=data[::-1]; y=np.arange(len(data))
    vals=np.array([d[1] for d in data])
    ax.axvspan(0,0.005,color="#FBF4DE",zorder=0)
    ax.axvline(0.005,color=GOLD,lw=0.75,ls="--")
    ax.hlines(y,0,vals,color=GREEN,lw=1.4)
    ax.scatter(vals,y,s=27,color=GREEN,edgecolor=WHITE,lw=0.5,zorder=3)
    ax.set_yticks(y,[d[0] for d in data])
    ax.set_xlim(-0.002,0.12); ax.set_xlabel("Oriented validation Δ")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",color=PALE,lw=0.7)
    ax.tick_params(axis="y",length=0)
    for yi,(name,v,n) in zip(y,data):
        suffix=f"  n={n:,}" if n else ""
        ax.text(v+0.002,yi,f"{v:+.3f}{suffix}",va="center",fontsize=6.0,color=INK)
    ax.text(0,-0.18,"Fixed recipe; discovery validation; external rows after exact filtering",
            transform=ax.transAxes,fontsize=5.5,color=GREY)

    # c
    ax=fig.add_subplot(gs[1,0]); panel(ax,"c","Independent confirmation revises memory")
    eps=sorted(confirm["episodes"],key=lambda e:e["mean_oriented_delta"])
    labels=[e["task"].replace("_AqSolDB","").replace("_AZ","").replace("_Veith","")
            .replace("Clearance_Microsome","Microsomal CL") for e in eps]
    vals=np.array([e["mean_oriented_delta"] for e in eps])
    colors=[{"supported":GREEN,"refuted":ORANGE,"inconclusive":GOLD}[e["status"]] for e in eps]
    y=np.arange(len(eps))
    ax.axvspan(-0.005,0.005,color="#FBF4DE",zorder=0)
    ax.axvline(0,color=INK,lw=0.65)
    ax.hlines(y,0,vals,color=[mpl.colors.to_rgba(c,0.55) for c in colors],lw=1.3)
    ax.scatter(vals,y,c=colors,s=27,edgecolor=WHITE,lw=0.5,zorder=3)
    ax.set_yticks(y,labels); ax.tick_params(axis="y",length=0)
    ax.set_xlim(-0.035,0.285); ax.set_xlabel("Confirmed oriented Δ")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",color=PALE,lw=0.7)
    for yi,v in zip(y,vals):
        ax.text(v+0.006,yi,f"{v:+.3f}",ha="left",va="center",fontsize=5.9,color=INK)
    ax.text(0,-0.18,"Pruning regret 0.259 → global negative-memory pruning withdrawn",
            transform=ax.transAxes,fontsize=5.6,color=ORANGE,fontweight="bold")

    # d: Pareto view
    ax=fig.add_subplot(gs[1,1]); panel(ax,"d","Performance–compute feedback")
    ax.axhspan(-0.005,0.005,color="#EAF5F1",zorder=0)
    ax.axhline(0,color=INK,lw=0.65)
    ax.axhline(-0.005,color=GREEN,lw=0.65,ls="--")
    mean=efficiency["mean_oriented_gain"]
    lo,hi=efficiency["bootstrap_95_ci"]
    cost=efficiency["aggregate_cost_ratio"]
    ax.errorbar([cost],[mean],yerr=[[mean-lo],[hi-mean]],fmt="o",ms=5,
                color=BLUE,ecolor=BLUE,capsize=3,zorder=3)
    ax.scatter([1],[0],s=30,marker="s",facecolor=WHITE,edgecolor=GREY,lw=0.9,zorder=3)
    ax.plot([cost,1],[mean,0],color=LIGHT_GREY,lw=0.8,zorder=1)
    ax.text(cost+0.03,mean+0.0009,
            f"Agent\n{cost:.3f}× compute\nΔ={mean:+.4f}",
            fontsize=6.1,color=BLUE,ha="left",va="bottom")
    ax.text(0.98,0.0006,"Score-only\n1.00×",fontsize=6.0,color=GREY,ha="right",va="bottom")
    ax.text(0.02,-0.00455,"practical non-inferiority region",fontsize=5.6,color=GREEN)
    ax.set_xlim(0,1.05); ax.set_ylim(-0.0062,0.0035)
    ax.set_xlabel("Relative selected-recipe compute")
    ax.set_ylabel("Oriented performance Δ")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color=PALE,lw=0.7)
    ax.text(0,-0.18,"95% CI shown; estimated compute reduction 72.2%",
            transform=ax.transAxes,fontsize=5.6,color=GREY)

    save(fig,"figure2_nature_multi_axis_feedback")


if __name__=="__main__":
    setup()
    figure1()
    figure2()
    print(OUT)
