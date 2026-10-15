# -*- coding: utf-8 -*-
"""数据溯源：把所有论文/PPT 关键数字从原始 json 现算一遍（可复现）。"""
import json, numpy as np, collections

sr = json.load(open("stage_results.json", encoding="utf-8"))
sp_s = json.load(open("benchmark_results.json", encoding="utf-8"))
sp_d = json.load(open("benchmark_results_dual.json", encoding="utf-8"))

print("=" * 66)
print("B. 机制类（stage_results.json = 机制探针逐局记录）")


def counter(pref, tids, cam):
    c = collections.Counter()
    for t in tids:
        v = sr.get(f"baseline|{cam}|None|{pref}{t}")
        if not v:
            continue
        for e in v["episodes"]:
            c[e["label"]] += 1
    return c


def n_ep(pref, tids, cam):
    return sum(counter(pref, tids, cam).values())


# B1 抓取失败（grasp_fail 局数）
cs = counter("task", range(10), "single")
cd = counter("task", range(10), "dual")
gs, gd = cs["grasp_fail"], cd["grasp_fail"]
tot = n_ep("task", range(10), "single")
print(f"  B1 抓取失败（Spatial 10 任务 × 50 局 = {tot} 局）")
print(f"     单相机 grasp_fail = {gs} 局 ({gs/tot*100:.1f}%) → 双相机 {gd} 局 ({gd/tot*100:.1f}%)")
print(f"     下降 = {(gs-gd)/gs*100:.1f}%  ← 论文「抓取失败 −74%」")

# B2 灾难性脱靶（闭爪距离 > 8cm）
def off_rate(pref, tids, cam, thr=0.08):
    tot = 0
    off = 0
    for t in tids:
        v = sr.get(f"baseline|{cam}|None|{pref}{t}")
        if not v:
            continue
        ds = [e["dist_close"] for e in v["episodes"] if e.get("dist_close") is not None]
        tot += len(ds)
        off += sum(1 for x in ds if x > thr)
    return off / tot * 100, off, tot

os_, ooff, otot = off_rate("task", range(10), "single")
od_, doff, dtot = off_rate("task", range(10), "dual")
print(f"  B2 灾难性脱靶（闭爪时末端-物体距离 > 8cm）")
print(f"     单相机 {ooff}/{otot} = {os_:.1f}% → 双相机 {doff}/{dtot} = {od_:.1f}%")
print(f"     下降 = {(os_-od_)/os_*100:.1f}%  ← 论文「脱靶率下降 66%」")

# B3 Object 链条断裂（no_grasp + grasp_fail + carry_fail）
CHAIN = ("no_grasp", "grasp_fail", "carry_fail")
ocs = counter("object_task", [0, 3, 6, 9], "single")
ocd = counter("object_task", [0, 3, 6, 9], "dual")
fs = sum(ocs[k] for k in CHAIN)
fd = sum(ocd[k] for k in CHAIN)
tot_o = n_ep("object_task", [0, 3, 6, 9], "single")
print(f"  B3 Object 链条断裂失败（{tot_o} 局）")
print(f"     单相机 {fs} 局 ({fs/tot_o*100:.0f}%) → 双相机 {fd} 局 ({fd/tot_o*100:.0f}%)")
for t in (0, 3, 6, 9):
    a = counter("object_task", [t], "single")
    b = counter("object_task", [t], "dual")
    fa = sum(a[k] for k in CHAIN)
    fb = sum(b[k] for k in CHAIN)
    print(f"       t{t}: {fa}/50 ({fa/50*100:.0f}%) → {fb}/50 ({fb/50*100:.0f}%)  "
          f"成功率 {sr[f'baseline|single|None|object_task{t}']['rate']:.0f}→"
          f"{sr[f'baseline|dual|None|object_task{t}']['rate']:.0f}%")

# B4 瞄准-增益（闭爪距离均值差 vs 增益）
xs, ys, pts = [], [], []
for t in range(10):
    a = sr[f"baseline|single|None|task{t}"]["episodes"]
    b = sr[f"baseline|dual|None|task{t}"]["episodes"]
    da = [e["dist_close"] for e in a if e.get("dist_close") is not None]
    db = [e["dist_close"] for e in b if e.get("dist_close") is not None]
    if da and db:
        d = (np.mean(db) - np.mean(da)) * 100
        xs.append(d)
        ys.append(sp_d[f"task{t}"]["rate"] - sp_s[f"task{t}"]["rate"])
        pts.append((t, d, ys[-1]))
print(f"  B4 瞄准：r(Δ闭爪距离, 增益) = {np.corrcoef(xs, ys)[0,1]:+.3f}  (n={len(xs)} 任务)")
print(f"     逐任务 Δ闭爪距离(cm): " + ", ".join(f"t{t}:{d:+.1f}" for t, d, _ in pts))

# B5 因果干预：冻结 vs 重放
print(f"  B5 因果干预（冻结 mask / 重放 mask-replay）")
rows = []
for k in sorted(k for k in sr if k.startswith("mask-replay|")):
    _, cam, stage, task = k.split("|")
    mk = f"mask|{cam}|{stage}|{task}"
    bk = f"baseline|{cam}|None|{task}"
    if mk not in sr or bk not in sr:
        continue
    B = sr[bk]["rate"]
    M = sr[mk]["rate"]
    R = sr[k]["rate"]
    dM, dR = B - M, B - R
    verdict = "伪影" if dR <= dM / 3 else ("真依赖" if dR > dM * 2 / 3 else "部分依赖")
    rows.append((task, stage, B, M, R, dM, dR, verdict))
    print(f"     {task:>7s} {stage:8s} 基线 {B:5.1f} | 冻结 {M:5.1f} (掉 {dM:5.1f}) | "
          f"重放 {R:5.1f} (掉 {dR:5.1f}) → {verdict}")
cnt = collections.Counter(r[-1] for r in rows)
print(f"     统计: {dict(cnt)}（共 {len(rows)} 格）")

print("=" * 66)
print("C. 开局级（init_pose_results.json）")
ip = json.load(open("init_pose_results.json", encoding="utf-8"))
by = collections.defaultdict(list)
for r in ip:
    by[(r["suite"], r["task"])].append(r)
print(f"  样本: {len(ip)} 行 = {len(by)} 任务 × {len(list(by.values())[0])} 初始态")
for suite in ("libero_spatial", "libero_object"):
    sub = [r for r in ip if r["suite"] == suite and r["d_goal"] is not None]
    x = np.array([r["d_goal"] for r in sub])
    y = np.array([r["ok_s"] for r in sub], dtype=float)
    print(f"  {suite}: n={len(sub)}, r(开局物体-容器距离, 单相机成败) = {np.corrcoef(x, y)[0,1]:+.3f}")
allx = np.array([r["d_goal"] for r in ip if r["d_goal"] is not None])
ally = np.array([r["ok_s"] for r in ip if r["d_goal"] is not None], dtype=float)
print(f"  混合 14 任务: r = {np.corrcoef(allx, ally)[0,1]:+.3f}  ← Simpson 混杂（远距样本均来自 Object）")
diffs = []
for (suite, t), sub in by.items():
    sub = [r for r in sub if r["d_goal"] is not None]
    med = np.median([r["d_goal"] for r in sub])
    lo = [r["ok_s"] for r in sub if r["d_goal"] <= med]
    hi = [r["ok_s"] for r in sub if r["d_goal"] > med]
    diffs.append((np.mean(lo) - np.mean(hi)) * 100)
print(f"  任务内「近半 − 远半」成功率差: |均值| {np.abs(diffs).mean():.1f}pp，"
      f"最大 {np.abs(diffs).max():.1f}pp，<20pp 的任务 {sum(1 for x in diffs if abs(x) < 20)}/{len(diffs)}")
tx, ty = [], []
for (suite, t), sub in by.items():
    sub = [r for r in sub if r["d_goal"] is not None]
    tx.append(np.mean([r["d_goal"] for r in sub]) * 100)
    ty.append(np.mean([r["ok_s"] for r in sub]) * 100)
print(f"  任务级 r(任务平均距离, 单相机成功率) = {np.corrcoef(tx, ty)[0,1]:+.3f} (n={len(tx)})")
