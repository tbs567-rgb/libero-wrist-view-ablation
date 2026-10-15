# -*- coding: utf-8 -*-
"""论文/答辩核心图表（大纲 12.x 图表化）：5 张。
输出：figs/fig1..fig5 .png（dpi 200）
数据源：benchmark_results*.json（Spatial）、object_check_results.json（Object）、
        init_pose_results.json（2.3）、stage_results.json（瞄准、taxo）。
用法：./.venv/Scripts/python.exe make_figs.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

C_POS = "#2b6cb0"   # 正增益 蓝
C_NEG = "#c53030"   # 负增益 红
C_SP = "#2b6cb0"    # Spatial 蓝
C_OB = "#dd6b20"    # Object 橙

SP_NAMES = ["0 盘盅之间", "1 烤盅旁", "2 桌中央", "3 盒上", "4 抽屉",
            "5 烤盅顶", "6 盒旁", "7 炉灶", "8 盘旁", "9 柜顶"]
OB_NAMES = {0: "汤罐", 3: "酱瓶", 6: "黄油", 9: "橙汁"}


def load_data():
    sp_s = json.load(open("benchmark_results.json", encoding="utf-8"))
    sp_d = json.load(open("benchmark_results_dual.json", encoding="utf-8"))
    ob = json.load(open("object_check_results.json", encoding="utf-8"))
    tasks = []
    for t in range(10):
        tasks.append({"key": f"sp{t}", "name": f"sp{t} {SP_NAMES[t]}",
                      "s": sp_s[f"task{t}"]["rate"], "d": sp_d[f"task{t}"]["rate"], "suite": "sp"})
    for t in (0, 3, 6, 9):
        tasks.append({"key": f"ob{t}", "name": f"ob{t} {OB_NAMES[t]}",
                      "s": ob[f"task{t}|single"]["rate"], "d": ob[f"task{t}|dual"]["rate"], "suite": "ob"})
    for tk in tasks:
        tk["g"] = tk["d"] - tk["s"]
    return tasks


def fig1_gain_spectrum(tasks):
    """横向斜率图：每任务一行（灰点=单相机 → 方块=双相机），标签在 y 轴不重叠。"""
    order = sorted(tasks, key=lambda x: x["s"])
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    for i, tk in enumerate(order):
        g_ = tk["g"]
        c = C_POS if g_ > 4.5 else (C_NEG if g_ < -4.5 else "#8B97A8")
        ax.plot([tk["s"], tk["d"]], [i, i], "-", color=c, lw=1.8, alpha=0.85, zorder=2)
        ax.scatter([tk["s"]], [i], s=30, color="#999", zorder=3)
        ax.scatter([tk["d"]], [i], s=48, marker="s", color=c, zorder=3)
        ax.annotate(f"{tk['g']:+.0f} pp", xy=(max(tk["s"], tk["d"]) + 1.5, i), fontsize=8.5,
                    va="center", color=c)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([tk["name"] for tk in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("官方协议成功率（%）", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("图 1 增益谱：14 任务单→双相机（按单相机基线升序；蓝=正增益 / 红=负增益 / 灰=±4pp 波动带）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_gain_spectrum.png"), dpi=200)
    plt.close(fig)


def fig2_baseline_vs_gain(tasks):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for suite, c, lab in (("sp", C_SP, "Spatial（10 任务）"), ("ob", C_OB, "Object（4 任务）")):
        xs = [tk["s"] for tk in tasks if tk["suite"] == suite]
        ys = [tk["g"] for tk in tasks if tk["suite"] == suite]
        ax.scatter(xs, ys, s=52, color=c, label=lab, zorder=3, edgecolors="white", linewidths=0.8)
    x = np.array([tk["s"] for tk in tasks]); y = np.array([tk["g"] for tk in tasks])
    r = np.corrcoef(x, y)[0, 1]
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 4, x.max() + 4, 50)
    ax.plot(xs, k * xs + b, "--", color="#555", lw=1.4, alpha=0.8)
    for tk in tasks:
        if abs(tk["g"]) >= 25 or tk["key"] in ("ob0",):
            ax.annotate(tk["name"], (tk["s"], tk["g"]), fontsize=8.5,
                        xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("单相机基线（%）", fontsize=11)
    ax.set_ylabel("腕部视角增益（pp）", fontsize=11)
    ax.set_title(f"图 2 单相机基线 vs 增益（14 任务）：r = {r:+.3f}", fontsize=11.5)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_baseline_vs_gain.png"), dpi=200)
    plt.close(fig)
    return r


def fig3_init_pose():
    rows = json.load(open("init_pose_results.json", encoding="utf-8"))
    units = {}
    for r in rows:
        k = (r["suite"], r["task"])
        units.setdefault(k, []).append(r)
    keys = sorted(units.keys(), key=lambda k: (k[0] != "libero_spatial", k[1]))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    # (a)(b): 任务级 d_goal vs 单 / 救援
    dg, s1, resc, dg_all, ok_all = [], [], [], [], []
    for k in keys:
        sub = units[k]
        dg.append(np.mean([r["d_goal"] for r in sub]) * 100)
        s1.append(np.mean([r["ok_s"] for r in sub]) * 100)
        resc.append(np.mean([bool(r["ok_d"]) and not bool(r["ok_s"]) for r in sub]) * 100)
        for r in sub:
            dg_all.append(r["d_goal"] * 100)
            ok_all.append(float(r["ok_s"]))
    dg = np.array(dg); s1 = np.array(s1); resc = np.array(resc)
    dg_all = np.array(dg_all); ok_all = np.array(ok_all)
    n_sp = sum(1 for k in keys if k[0] == "libero_spatial")

    for ax, yv, ylab, tt in ((axes[0], s1, "单相机成功率（%）", "任务级：距离 vs 单相机"),
                             (axes[1], resc, "救援率（%）", "任务级：距离 vs 救援率")):
        ax.scatter(dg[:n_sp], yv[:n_sp], s=46, color=C_SP, label="Spatial", zorder=3, edgecolors="white")
        ax.scatter(dg[n_sp:], yv[n_sp:], s=46, color=C_OB, label="Object", zorder=3, edgecolors="white")
        r = np.corrcoef(dg, yv)[0, 1]
        k2, b2 = np.polyfit(dg, yv, 1)
        xs = np.linspace(dg.min() - 2, dg.max() + 2, 40)
        ax.plot(xs, k2 * xs + b2, "--", color="#555", lw=1.3)
        ax.set_xlabel("任务平均 d_goal（cm）", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(f"{tt}  r = {r:+.2f}", fontsize=10.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8.5, loc="upper right")

    # (c) 任务内近/远半差
    gaps = []
    for k in keys:
        sub = units[k]
        vals = np.array([r["d_goal"] for r in sub])
        med = np.median(vals)
        lo = [r for r in sub if r["d_goal"] <= med]
        hi = [r for r in sub if r["d_goal"] > med]
        gaps.append((np.mean([r["ok_s"] for r in lo]) - np.mean([r["ok_s"] for r in hi])) * 100)
    gaps = np.array(gaps)
    order = np.argsort(gaps)
    names = [f"{keys[i][0][-7:]}·t{keys[i][1]}" for i in order]
    axes[2].bar(range(len(gaps)), gaps[order],
                color=[C_POS if g > 0 else C_NEG for g in gaps[order]], alpha=0.85)
    axes[2].axhline(0, color="#333", lw=0.9)
    axes[2].axhline(np.abs(gaps).mean(), color="#888", ls=":", lw=1)
    axes[2].set_xticks(range(len(gaps)))
    axes[2].set_xticklabels(names, rotation=90, fontsize=6.5)
    axes[2].set_ylabel("近半 − 远半（单相机，pp）", fontsize=10)
    axes[2].set_title(f"任务内局部分差（|均值|={np.abs(gaps).mean():.1f}pp，仅 2/14 超过 20pp）", fontsize=10.5)
    axes[2].grid(axis="y", alpha=0.3)
    fig.suptitle("图 3 开局级条件分析（n=700）：几何距离不是核心变量", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_init_pose.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return dg_all, ok_all


def fig4_object_failures():
    tx = json.load(open("object_taxo_results.json", encoding="utf-8"))
    classes = ["未接近", "接触未提起", "提起后未运输", "运输后未送达"]
    colors = ["#4a5568", "#dd6b20", "#c53030", "#805ad5"]
    tids = [0, 3, 6, 9]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width = 0.38
    xbase = np.arange(len(tids))
    for j, cam in enumerate(("single", "dual")):
        xpos = xbase + (j - 0.5) * width
        bottoms = np.zeros(len(tids))
        for ci, cl in enumerate(classes):
            vals = [tx[f"task{t}|{cam}"]["classes"].get(cl, 0) for t in tids]
            ax.bar(xpos, vals, width * 0.92, bottom=bottoms, color=colors[ci],
                   label=cl if j == 0 else None, alpha=0.92 if j == 1 else 0.75)
            bottoms += np.array(vals)
        for xi, t in zip(xpos, tids):
            tot = bottoms[list(xpos).index(xi)]
            ax.text(xi, tot + 0.6, f"{int(tot)}", ha="center", fontsize=8.5)
    ax.set_xticks(xbase)
    ax.set_xticklabels([f"{OB_NAMES[t]}\n(t{t})" for t in tids], fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", title="失败类型")
    ax.text(0.01, 0.985, "每任务：左柱=单相机  右柱=双相机", transform=ax.transAxes,
            fontsize=9, va="top", color="#333")
    ax.set_ylabel("失败局数（每格 50 局）", fontsize=10.5)
    ax.set_title("图 4 Object 失败画像：单相机的「抓取-运输链条断裂」被腕部视角清零（柱高=失败总数）", fontsize=10.5)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_object_failures.png"), dpi=200)
    plt.close(fig)


def fig5_aiming_gain():
    r = json.load(open("stage_results.json", encoding="utf-8"))
    xs, ys, names = [], [], []
    for t in range(10):
        try:
            es = [e["dist_close"] for e in r[f"baseline|single|None|task{t}"]["episodes"] if e.get("dist_close")]
            ed = [e["dist_close"] for e in r[f"baseline|dual|None|task{t}"]["episodes"] if e.get("dist_close")]
            if not es or not ed:
                continue
            delta = (np.mean(ed) - np.mean(es)) * 100  # 双-单（cm，负=双更准）
            g = r[f"baseline|dual|None|task{t}"]["rate"] - r[f"baseline|single|None|task{t}"]["rate"]
            xs.append(delta); ys.append(g); names.append(f"t{t}")
        except KeyError:
            continue
    xs = np.array(xs); ys = np.array(ys)
    rr = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(xs, ys, s=56, color=C_SP, zorder=3, edgecolors="white", linewidths=0.8)
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm, (x, y), fontsize=9, xytext=(5, 3), textcoords="offset points")
    k, b = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 40)
    ax.plot(xr, k * xr + b, "--", color="#555", lw=1.4)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("Δ闭爪瞄准误差（双−单，cm；负值=双相机更准）", fontsize=10.5)
    ax.set_ylabel("增益（pp）", fontsize=10.5)
    ax.set_title(f"图 5 瞄准改善 vs 增益（Spatial {len(xs)} 任务）：r = {rr:+.2f}", fontsize=11.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_aiming_gain.png"), dpi=200)
    plt.close(fig)
    return rr, list(zip(names, xs.round(2), ys))


def fig6_mask_intervention():
    """阶段干预全景：4 任务 × 4 阶段，冻结 vs 重放（+基线参考线）。"""
    r = json.load(open("stage_results.json", encoding="utf-8"))
    tids = [2, 9, 7, 8]
    stages = ["approach", "grasp", "carry", "place"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for ax, t in zip(axes.flat, tids):
        base = r[f"baseline|dual|None|task{t}"]["rate"]
        x = np.arange(len(stages))
        fr, rp = [], []
        for st in stages:
            k1, k2 = f"mask|dual|{st}|task{t}", f"mask-replay|dual|{st}|task{t}"
            fr.append(r[k1]["rate"] if k1 in r else np.nan)
            rp.append(r[k2]["rate"] if k2 in r else np.nan)
        w = 0.36
        ax.bar(x - w / 2, fr, w, color="#c53030", alpha=0.85, label="冻结 freeze")
        ax.bar(x + w / 2, rp, w, color="#38a169", alpha=0.85, label="重放 replay")
        ax.axhline(base, color="#2b6cb0", ls="--", lw=1.6, label=f"基线 {base:.0f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylabel("成功率（%）", fontsize=9.5)
        ax.set_ylim(0, 108)
        ax.set_title(f"task{t}（双相机基线 {base:.0f}%）", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("图 6 阶段干预：腕部视角「冻结 vs 重放」双对照（10 格对照中 8 格重放恢复 → 冻结跌幅多为伪影；place t7/t8 存在真依赖）", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_mask_intervention.png"), dpi=200)
    plt.close(fig)


def main():
    tasks = load_data()
    print("=== 14 任务数据 ===")
    for tk in tasks:
        print(f"  {tk['name']:16s} 单{tk['s']:5.1f} 双{tk['d']:5.1f}  增益{tk['g']:+5.1f}")
    fig1_gain_spectrum(tasks)
    r2 = fig2_baseline_vs_gain(tasks)
    fig3_init_pose()
    fig4_object_failures()
    rr5, pts5 = fig5_aiming_gain()
    fig6_mask_intervention()
    print(f"\n[图2] r(基线, 增益) = {r2:+.3f}（论文引用 −0.94）")
    print(f"[图5] r(Δ瞄准, 增益) = {rr5:+.3f}（论文引用 −0.72）")
    print("[图5] 各任务 Δ瞄准:", pts5)
    print("\n输出目录:", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
