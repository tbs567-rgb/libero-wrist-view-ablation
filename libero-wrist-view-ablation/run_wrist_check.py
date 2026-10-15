# -*- coding: utf-8 -*-
"""
仅腕部（wrist-only）对照实验：三组对照（仅全局 / 仅腕部 / 双视角）补全第三组。
设计参照 Look Closer（RA-L 2022）的 single-third-person / single-egocentric / dual-fusion 三分法，
但在我们的 BC 管线内以"唯一变量"方式实现（同一网络 n_cam=1，仅数据源换成 eye_in_hand）。

任务选 3 个代表：spatial t2（旗舰正增益 +56）、spatial t7（负增益代表）、object t9（橙汁 6→94）。
流程（断点续跑）：每任务 train wrist(600ep) → eval wrist(50 局, 256) → 写 wrist_check_results.json。
汇总时与已有 single/dual 结果（benchmark_results*.json / object_check_results.json）拼三柱表。
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if os.environ.get("_LIBERO_REEXEC") != "1" and os.path.isfile(VENV):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV):
        env = dict(os.environ, _LIBERO_REEXEC="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        raise SystemExit(subprocess.call([VENV, os.path.abspath(__file__)] + sys.argv[1:], env=env))

import libero_env as LE  # noqa: E402,F401

UNITS = [("libero_spatial", 2), ("libero_spatial", 7), ("libero_object", 9)]
EPOCHS = 600
EPISODES = 50
RESULTS = os.path.join(HERE, "wrist_check_results.json")
MODEL_DIR = os.path.join(HERE, "low_resource_models")
EVAL_ENV = dict(os.environ, LIBERO_EVAL_CAM="256", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=HERE, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    res = {}
    if os.path.isfile(RESULTS):
        res = json.load(open(RESULTS, encoding="utf-8"))
    for suite_name, tid in UNITS:
        suite = LE.get_suite(suite_name)
        t = suite.get_task(tid)
        h5 = os.path.join(LE.get_libero_path("datasets"), suite_name,
                          t.language.replace(" ", "_") + "_demo.hdf5")
        if not os.path.isfile(h5):
            print(f"[跳过] 数据缺失: {h5}", flush=True)
            continue
        key = f"{suite_name}|task{tid}|wrist"
        mp = os.path.join(MODEL_DIR, "official_resnet_wrist_"
                          + os.path.basename(h5).replace("_demo.hdf5", ".pt"))
        # ---- 训练（已有模型则跳过）----
        if not os.path.isfile(mp):
            cmd = [VENV, "low_resource_official.py", "train", "--h5", h5,
                   "--epochs", str(EPOCHS), "--wrist-cam"]
            print(f"===== [train wrist] {suite_name} task{tid}: {t.language[:50]} =====", flush=True)
            code, out = run(cmd)
            print(out[-400:], flush=True)
            if code != 0 or not os.path.isfile(mp):
                print(f"[训练失败] {suite_name} task{tid} wrist", flush=True)
                continue
        # ---- 评测 ----
        cmd = [VENV, "low_resource_official.py", "eval", "--h5", h5, "--wrist-cam",
               "--episodes", str(EPISODES), "--task-id", str(tid), "--suite", suite_name]
        print(f"===== [eval wrist] {suite_name} task{tid} =====", flush=True)
        code, out = run(cmd, env=EVAL_ENV)
        line = [l for l in out.splitlines() if "官方协议成功率" in l]
        if not line:
            print(f"[评测失败] {suite_name} task{tid} wrist\n{out[-300:]}", flush=True)
            continue
        succ = int(line[-1].split(":")[-1].strip().split("/")[0])
        res[key] = {"suite": suite_name, "task": tid, "lang": t.language,
                    "succ": succ, "n": EPISODES, "rate": round(succ / EPISODES * 100, 1)}
        json.dump(res, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[完成] {key}: {succ}/{EPISODES} = {res[key]['rate']}%", flush=True)

    # ---- 三柱汇总（single/dual 取自既有权威结果）----
    print("\n===== 三组对照汇总（仅全局 / 仅腕部 / 双视角）=====", flush=True)
    bm_s = json.load(open(os.path.join(HERE, "benchmark_results.json"), encoding="utf-8"))
    bm_d = json.load(open(os.path.join(HERE, "benchmark_results_dual.json"), encoding="utf-8"))
    ob = json.load(open(os.path.join(HERE, "object_check_results.json"), encoding="utf-8"))
    for suite_name, tid in UNITS:
        key = f"{suite_name}|task{tid}|wrist"
        if suite_name == "libero_spatial":
            single = bm_s[f"task{tid}"]["rate"]; dual = bm_d[f"task{tid}"]["rate"]
        else:
            single = ob[f"task{tid}|single"]["rate"]; dual = ob[f"task{tid}|dual"]["rate"]
        w = res.get(key, {}).get("rate")
        lang = res.get(key, {}).get("lang", "")[:40]
        wtxt = f"{w}%" if w is not None else "未完成"
        print(f"  {suite_name} task{tid} ({lang}): 单={single}%  仅腕部={wtxt}  双={dual}%", flush=True)


if __name__ == "__main__":
    main()
