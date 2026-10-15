# -*- coding: utf-8 -*-
"""
Object 套件跨套件抽检：4 任务 × 单/双相机 = 8 训练-评测单元。
任务：0 字母汤 / 3 烧烤酱 / 6 黄油 / 9 橙汁（物体形态各异）
目的：验证 Spatial 上的机制结论（增益集中于抓取失败修复、遮挡否定）在 Object 套件是否成立。
流程（断点续跑，会自动跳过已完成的模型/结果）：
  每个任务：train single → eval single(50局,256) → train dual → eval dual(50局,256)
结果写 object_check_results.json。
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

SUITE = "libero_object"
TASKS = [0, 3, 6, 9]
EPOCHS = 600
EPISODES = 50
RESULTS = os.path.join(HERE, "object_check_results.json")
MODEL_DIR = os.path.join(HERE, "low_resource_models")
EVAL_ENV = dict(os.environ, LIBERO_EVAL_CAM="256", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=HERE, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    suite = LE.get_suite(SUITE)
    res = {}
    if os.path.isfile(RESULTS):
        res = json.load(open(RESULTS, encoding="utf-8"))
    dsdir = os.path.join(LE.get_libero_path("datasets"), SUITE)
    for tid in TASKS:
        t = suite.get_task(tid)
        h5 = os.path.join(dsdir, t.language.replace(" ", "_") + "_demo.hdf5")
        if not os.path.isfile(h5):
            print(f"[跳过] 数据缺失: {h5}", flush=True)
            continue
        for dual in (False, True):
            tag = "dual" if dual else "single"
            key = f"task{tid}|{tag}"
            mp = os.path.join(MODEL_DIR, ("official_resnet_dual_" if dual else "official_resnet_")
                              + os.path.basename(h5).replace("_demo.hdf5", ".pt"))
            # ---- 训练（已有模型则跳过）----
            if not os.path.isfile(mp):
                cmd = [VENV, "low_resource_official.py", "train", "--h5", h5, "--epochs", str(EPOCHS)]
                if dual:
                    cmd.append("--dual-cam")
                print(f"===== [train {tag}] task{tid}: {t.language[:50]} =====", flush=True)
                code, out = run(cmd)
                print(out[-400:], flush=True)
                if code != 0 or not os.path.isfile(mp):
                    print(f"[训练失败] task{tid} {tag}", flush=True)
                    continue
            # ---- 评测 ----
            cmd = [VENV, "low_resource_official.py", "eval", "--h5", h5, "--episodes", str(EPISODES),
                   "--task-id", str(tid), "--suite", SUITE]
            if dual:
                cmd.append("--dual-cam")
            print(f"===== [eval {tag}] task{tid} =====", flush=True)
            code, out = run(cmd, env=EVAL_ENV)
            line = [l for l in out.splitlines() if "官方协议成功率" in l]
            if not line:
                print(f"[评测失败] task{tid} {tag}\n{out[-300:]}", flush=True)
                continue
            succ = int(line[-1].split(":")[-1].strip().split("/")[0])
            res[key] = {"task": tid, "lang": t.language, "dual": dual,
                        "succ": succ, "n": EPISODES, "rate": round(succ / EPISODES * 100, 1)}
            json.dump(res, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[完成] {key}: {succ}/{EPISODES} = {res[key]['rate']}%", flush=True)

    print("\n===== Object 跨套件抽检汇总 =====", flush=True)
    gains = []
    for tid in TASKS:
        s = res.get(f"task{tid}|single", {}).get("rate")
        d = res.get(f"task{tid}|dual", {}).get("rate")
        if s is not None and d is not None:
            gains.append(d - s)
            print(f"  task{tid} ({res[f'task{tid}|single']['lang'][:40]}): "
                  f"single={s}%  dual={d}%  增益={d-s:+.1f}pp", flush=True)
        else:
            print(f"  task{tid}: 未完成", flush=True)
    if gains:
        print(f"\n  Object 4 任务平均增益: {sum(gains)/len(gains):+.1f}pp（Spatial 为 +17.0pp）", flush=True)


if __name__ == "__main__":
    main()
