# -*- coding: utf-8 -*-
"""多种子复现批次（研究大纲 5.1）：Spatial t2 / t7 × 单/双 × seed∈{1,2}（seed7 为既有基线）。

每单元：600 epoch 训练（模型文件已存在则跳过）→ 256 渲染官方协议 50 局评测。
结果增量写入 multiseed_results.json；结束后打印 3 种子（含 seed7）均值±标准差。

用法（后台）：
  ./.venv/Scripts/python.exe -u run_multiseed.py
"""
import os
import re
import json
import time
import statistics
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv", "Scripts", "python.exe")
MODEL_DIR = os.path.join(HERE, "low_resource_models")
RESULTS = os.path.join(HERE, "multiseed_results.json")
DATASET = os.path.join(HERE, "datasets", "libero_spatial")

H5S = {
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5",
}
TASKS = [2, 7]
SEEDS = [1, 2]
EPOCHS, EPISODES = 600, 50
EVAL_ENV = dict(os.environ, LIBERO_EVAL_CAM="256", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def key_of(tid, dual, seed):
    return f"task{tid}|{'dual' if dual else 'single'}|s{seed}"


def model_path(tid, dual, seed):
    base = H5S[tid].replace("_demo.hdf5", "")
    tag = "_dual" if dual else ""
    seed_tag = "" if seed == 7 else f"_s{seed}"
    return os.path.join(MODEL_DIR, f"official_resnet{tag}{seed_tag}_{base}.pt")


def run(cmd, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0


def main():
    res = json.load(open(RESULTS, encoding="utf-8")) if os.path.isfile(RESULTS) else {}
    units = [(t, d, s) for t in TASKS for d in (False, True) for s in SEEDS]
    t_all = time.time()
    for attempt in (1, 2):
        pending = [u for u in units if key_of(*u) not in res]
        if not pending:
            break
        if attempt == 2:
            print("第二遍重试未完成单元 ...", flush=True)
        for idx, (tid, dual, seed) in enumerate(pending, 1):
            tag = key_of(tid, dual, seed)
            h5 = os.path.join(DATASET, H5S[tid])
            mp = model_path(tid, dual, seed)
            print(f"[{idx}/{len(pending)}] {tag} ====", flush=True)
            if not os.path.isfile(mp):
                cmd = [VENV, "low_resource_official.py", "train", "--h5", h5,
                       "--epochs", str(EPOCHS), "--seed", str(seed)]
                if dual:
                    cmd.append("--dual-cam")
                print("  训练中 ...", flush=True)
                code, out, dt = run(cmd)
                if code != 0 or not os.path.isfile(mp):
                    print(f"  [失败] 训练 code={code}\n{out[-400:]}", flush=True)
                    continue
                print(f"  训练完成 ({dt/60:.0f}min): " + (out.splitlines()[-1] if out else ""), flush=True)
            else:
                print("  模型已存在，跳过训练", flush=True)
            cmd = [VENV, "low_resource_official.py", "eval", "--h5", h5,
                   "--episodes", str(EPISODES), "--task-id", str(tid), "--seed", str(seed)]
            if dual:
                cmd.append("--dual-cam")
            print("  评测中 ...", flush=True)
            code, out, dt = run(cmd, env=EVAL_ENV)
            m = re.search(r"官方协议成功率:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", out)
            if code == 0 and m:
                succ, n, rate = int(m.group(1)), int(m.group(2)), float(m.group(3))
                res[tag] = {"succ": succ, "n": n, "rate": rate, "render": 256}
                json.dump(res, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"  完成 {tag}: {succ}/{n} = {rate}%  ({dt/60:.0f}min)", flush=True)
            else:
                print(f"  [失败] 评测 code={code}: {out[-300:]}", flush=True)

    print("\n===== 多种子汇总（seed7 基线 + 新种子）=====", flush=True)
    sing = json.load(open(os.path.join(HERE, "benchmark_results.json"), encoding="utf-8"))
    dualr = json.load(open(os.path.join(HERE, "benchmark_results_dual.json"), encoding="utf-8"))
    for tid in TASKS:
        row = {}
        for twin, ref in (("single", sing), ("dual", dualr)):
            rates = [ref[f"task{tid}"]["rate"]]
            for seed in SEEDS:
                k = key_of(tid, twin == "dual", seed)
                if k in res:
                    rates.append(res[k]["rate"])
            mu = statistics.mean(rates)
            sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            row[twin] = (rates, mu, sd)
            print(f"  task{tid} {twin:6s}: {rates} → {mu:.1f} ± {sd:.1f} (n={len(rates)})", flush=True)
        if all(len(v[0]) >= 3 for v in row.values()):
            print(f"  task{tid} 增益: {row['dual'][1] - row['single'][1]:+.1f}pp（3 种子均值）", flush=True)
    print(f"总耗时 {(time.time() - t_all) / 3600:.1f} 小时", flush=True)
    print("==== MULTISEED DONE ====", flush=True)


if __name__ == "__main__":
    main()
