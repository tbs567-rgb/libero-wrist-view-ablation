# -*- coding: utf-8 -*-
"""2.3 初始位姿级分析（n=700，零 GPU 推理成本）。

对 14 个任务（Spatial 10 + Object 4）提取每个初始状态（各 50 个）的：
  · 目标物坐标（obj_xy / obj_z）
  · 容器（盘子/篮子）坐标 → 目标-容器距离 d_goal
  · 夹爪初始位置 → 目标-夹爪初始距离 d_eef
再与机制探针记录的逐局成败（单/双相机，同初始态配对）联合分析：
  · 按 d_goal 分档 → 单/双成功率与救援率
  · 相关系数：r(d_goal, 单成功) / r(d_goal, 双成功) / r(d_goal, 救援)
回答：腕部视角在"什么样的开局"上最有用（把任务级条件规律细化到开局级）。

输出：init_pose_results.json + 屏幕汇总。
用法：./.venv/Scripts/python.exe -u init_pose_analysis.py
"""
import libero_env  # noqa: F401  必须最先导入
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "stage_results.json")
OUT = os.path.join(HERE, "init_pose_results.json")


def main():
    import stage_analysis as SA

    stage = json.load(open(STAGE, encoding="utf-8"))
    units = [("libero_spatial", t) for t in range(10)] + \
            [("libero_object", t) for t in [0, 3, 6, 9]]

    rows = []
    for suite_name, tid in units:
        SA.SUITE = suite_name
        env, suite, task = SA.build_env(tid, 64)
        inits = suite.get_task_init_states(tid)
        n = min(50, len(inits))

        env.reset()
        obs0 = env.set_init_state(inits[0])
        tgt_sub = SA.auto_target_substr(task.language, obs0)
        goal_sub = SA.auto_goal_substr(task.language)
        tk = [k for k in obs0 if tgt_sub in k and k.endswith("_pos") and "_to_robot0" not in k]
        assert tk, f"{suite_name} t{tid}: 找不到目标键 ({tgt_sub})"
        tkey = tk[0]

        def ok_list(cam):
            prefix = "object_task" if suite_name != "libero_spatial" else "task"
            v = stage.get(f"baseline|{cam}|None|{prefix}{tid}", {})
            arr = [None] * n
            for e in sorted(v.get("episodes", []), key=lambda x: x["ep"]):
                if e["ep"] < n:
                    arr[e["ep"]] = bool(e["success"])
            return arr

        okS, okD = ok_list("single"), ok_list("dual")

        for i in range(n):
            env.reset()
            obs = env.set_init_state(inits[i])
            obj = np.asarray(obs[tkey], dtype=float)
            goal = SA.find_goal_xy(env, obs, goal_sub)
            eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
            d_goal = None
            gxy = None
            if goal is not None:
                gxy = [round(float(goal[0]), 4), round(float(goal[1]), 4)]
                d_goal = round(float(np.linalg.norm(obj[:2] - np.asarray(goal, dtype=float)[:2])), 4)
            rows.append({
                "suite": suite_name, "task": tid, "init": i,
                "ok_s": okS[i], "ok_d": okD[i],
                "obj_xy": [round(float(obj[0]), 4), round(float(obj[1]), 4)],
                "obj_z": round(float(obj[2]), 4),
                "goal_xy": gxy,
                "d_goal": d_goal,
                "d_eef": round(float(np.linalg.norm(obj - eef)), 4),
            })
        env.close()
        print(f"[{suite_name} task{tid}] 完成（{n} 个初始位姿）", flush=True)

    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已保存: {OUT}（共 {len(rows)} 行）", flush=True)

    # ---------------- 分析 ----------------
    def arr(name):
        vals = [r[name] for r in rows if r[name] is not None]
        return np.array(vals, dtype=float)

    valid = [r for r in rows if r["d_goal"] is not None and r["ok_s"] is not None and r["ok_d"] is not None]
    print(f"有效行数（有 d_goal 与成败）: {len(valid)}", flush=True)

    def stats(sub, label):
        if not sub:
            return
        s = np.mean([r["ok_s"] for r in sub]) * 100
        d = np.mean([r["ok_d"] for r in sub]) * 100
        rescue = np.mean([(r["ok_d"] and not r["ok_s"]) for r in sub]) * 100
        broken = np.mean([(r["ok_s"] and not r["ok_d"]) for r in sub]) * 100
        print(f"  {label:44s} n={len(sub):3d}  单{s:5.1f}%  双{d:5.1f}%  救援{rescue:5.1f}%  破坏{broken:5.1f}%", flush=True)

    print("\n=== 总体与分套件 ===", flush=True)
    stats(valid, "全部")
    stats([r for r in valid if r["suite"] == "libero_spatial"], "Spatial")
    stats([r for r in valid if r["suite"] == "libero_object"], "Object")

    print("\n=== 按 目标-容器距离 d_goal 四分位分档（全部数据）===", flush=True)
    dg = np.array([r["d_goal"] for r in valid])
    qs = np.quantile(dg, [0.25, 0.5, 0.75])
    edges = [-1] + list(qs) + [999]
    for j in range(4):
        lo, hi = edges[j], edges[j + 1]
        sub = [r for r in valid if lo < r["d_goal"] <= hi]
        stats(sub, f"d_goal ∈ ({lo*100:.0f}, {hi*100:.0f}] cm")

    print("\n=== 相关系数（d_goal vs 成败；全部 + 分套件）===", flush=True)
    for lbl, sub in (("全部", valid),
                     ("Spatial", [r for r in valid if r["suite"] == "libero_spatial"]),
                     ("Object", [r for r in valid if r["suite"] == "libero_object"])):
        if len(sub) < 10:
            continue
        d = np.array([r["d_goal"] for r in sub])
        s = np.array([float(r["ok_s"]) for r in sub])
        dd = np.array([float(r["ok_d"]) for r in sub])
        rs = np.array([float(bool(r["ok_d"]) and not bool(r["ok_s"])) for r in sub])
        print(f"  {lbl:8s}: r(d_goal,单)={np.corrcoef(d, s)[0,1]:+.3f}  "
              f"r(d_goal,双)={np.corrcoef(d, dd)[0,1]:+.3f}  "
              f"r(d_goal,救援)={np.corrcoef(d, rs)[0,1]:+.3f}", flush=True)

    print("\n=== 各任务 d_goal 取值范围（cm）===", flush=True)
    for suite_name, tid in units:
        sub = [r for r in valid if r["suite"] == suite_name and r["task"] == tid]
        if not sub:
            continue
        d = np.array([r["d_goal"] for r in sub])
        z = np.array([r["obj_z"] for r in sub])
        print(f"  {suite_name[-7:]:>7s} t{tid}: d_goal {d.min()*100:5.1f} ~ {d.max()*100:5.1f}（跨度 {(d.max()-d.min())*100:4.1f}） "
              f" obj_z {z.min()*100:4.1f}~{z.max()*100:4.1f}", flush=True)

    print("\nINIT POSE ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
