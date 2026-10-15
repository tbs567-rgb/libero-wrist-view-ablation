# -*- coding: utf-8 -*-
"""
③ 阶段屏蔽反事实 + ④ 失败模式分类（带探针的评测）

- baseline 模式：不改输入，记录每回合的阶段指标，用于正面机制分析：
    · 闭爪瞄准误差 dist_close / dist_close_xy（夹爪闭合瞬间与碗的距离）
    · 闭爪前推碗量 push_at_close（接近过程是否把碗推歪）
    · 是否抬起 / 是否进入盘区 / 回合末碗距盘心 plate_dist_end
    · 失败标签：no_grasp / grasp_fail / carry_fail / place_fail
- mask 模式：在指定阶段把腕部视角"冻结"（窗口开启时刻起重复同一帧），测成功率跌幅（因果定位）。

阶段状态机（在线检测）：
  approach --(夹爪闭合 g1<0.006)--> grasp --(碗抬升>0.03m)--> carry --(碗进入盘区 xy<0.08m)--> place

用法：
  python stage_analysis.py baseline --tasks 0,1,...,9 --dual
  python stage_analysis.py baseline --suite libero_object --tasks 0,3,6,9 --dual
  python stage_analysis.py mask --tasks 2,9,7,8 --stage grasp
结果增量保存到 stage_results.json（object 套件键含 object_ 前缀，与 spatial 不冲突）。
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
VENV = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if os.environ.get("_LIBERO_REEXEC") != "1" and os.path.isfile(VENV):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV):
        env = dict(os.environ, _LIBERO_REEXEC="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        raise SystemExit(subprocess.call([VENV, os.path.abspath(__file__)] + sys.argv[1:], env=env))

import libero_env as LE  # noqa: E402  (mujoco 必须先于 torch)
import numpy as np  # noqa: E402
from collections import deque  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from low_resource_bc import image, make_state, MAX_STEPS, MODEL_DIR  # noqa: E402
from low_resource_resnet_fs import ResNetFSBC  # noqa: E402
from low_resource_framestack import K  # noqa: E402

SUITE = "libero_spatial"
IN_CAM = 96
RENDER = int(os.environ.get("LIBERO_EVAL_CAM", "256"))
RESULT_FILE = os.path.join(HERE, "stage_results.json")
EPISODES = 50
LIFT_THR = 0.03          # 碗相对初始抬升超过 3cm 记为"已抬起"
PLATE_R = 0.08           # 碗 xy 距盘心小于 8cm 记为"已进入盘区"
G_CLOSED = 0.006         # 手指 1 位置低于此值记为"已闭合"（张开~0.04，闭合~0.001）
G_OPEN = 0.02            # 高于此值记为"重开"（仅作指标记录，本模型多数不松爪）


def build_env(task_id, cam):
    suite = LE.get_suite(SUITE)
    task = suite.get_task(task_id)
    bddl = os.path.join(LE.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = LE.OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"], camera_heights=cam, camera_widths=cam)
    env.seed(0)
    return env, suite, task


def model_path_for(h5, dual):
    tag = "_dual" if dual else ""
    return os.path.join(MODEL_DIR, f"official_resnet{tag}_" + os.path.basename(h5).replace("_demo.hdf5", ".pt"))


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    n_cam = int(ckpt.get("n_cam", 1))
    model = ResNetFSBC(state_dim=8, n_cam=n_cam).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, n_cam


def auto_goal_substr(language):
    """从任务语言推断目标容器键名子串（plate / basket / …）。"""
    lang = (language or "").lower()
    for marker in ("place it in the ", "place it on top of the ", "place it on the "):
        if marker in lang:
            return lang.split(marker, 1)[1].strip().split(" ")[0]
    return "plate"


def auto_target_substr(language, obs):
    """从任务语言推断被操作物体的键名子串。

    - object 套件: 'pick up the alphabet soup and place it in the basket' → 'alphabet_soup'
    - spatial 套件: 描述含位置介词短语（on/between/next/in…）或 obs 中无匹配键 → 默认黑碗。
    """
    lang = (language or "").lower()
    if "pick up the " in lang:
        tail = lang.split("pick up the ", 1)[1]
        for sep in (" and place", " and put"):
            if sep in tail:
                seg = tail.split(sep, 1)[0].strip().replace(" ", "_")
                simple = all(tok not in seg for tok in ("_on_", "_between_", "_in_", "_next_"))
                if simple and any(seg in k and "_to_robot0" not in k for k in obs):
                    return seg
                break
    return "akita_black_bowl"


def find_goal_xy(env, obs, goal_substr="plate"):
    """优先从 obs 里找目标容器位置键（plate/basket）；找不到则从 sim body 读取（容器为静态物体，读一次即可）。"""
    plate_keys = [k for k in obs if goal_substr in k and k.endswith("_pos") and "_to_robot0" not in k]
    if plate_keys:
        return np.asarray(obs[plate_keys[0]], dtype=float)[:2]
    try:
        import mujoco
        env.sim.forward()
        m = env.sim.model._model            # reset 后必须重取引用
        d = env.sim.data._data
        for b in range(m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if goal_substr in n and n.endswith("_main"):
                return np.asarray(d.xpos[b][:2], dtype=float).copy()
    except Exception:
        pass
    return None


def run_episode(env, inits, idx, model, n_cam, device, mask_stage=None,
                mask_mode="freeze", replay_frames=None, record_wrist=None, task_language=None):
    """执行一个回合，返回指标 dict。mask_stage in {None,'approach','grasp','carry','place'}
    mask_mode: freeze=冻结为窗口开启帧；replay=重放另一回合的真实画面（解耦但动态）。
    record_wrist: 传入 list 则记录本回合每一步的腕部帧（用于后续 replay）。
    task_language: 任务语言（用于自动推断目标物体与目标容器；object 套件必需）。"""
    env.reset()
    obs = env.set_init_state(inits[idx])
    target_substr = auto_target_substr(task_language, obs) if task_language else "akita_black_bowl"
    goal_substr = auto_goal_substr(task_language) if task_language else "plate"
    tgt_keys = [k for k in obs if target_substr in k and k.endswith("_pos") and "_to_robot0" not in k]
    if not tgt_keys:
        raise RuntimeError(f"找不到目标物体键（substr='{target_substr}'）")
    target = tgt_keys[0]
    if idx == 0:
        print(f"  [探针] target='{target_substr}' → {target}   goal='{goal_substr}'", flush=True)
    bowl0 = np.asarray(obs[target], dtype=float)
    z0 = float(bowl0[2])
    plate_xy = find_goal_xy(env, obs, goal_substr)

    buf, buf2 = deque(maxlen=K), deque(maxlen=K)
    first = image(obs)
    if first.shape[0] != IN_CAM:
        first = cv2.resize(first, (IN_CAM, IN_CAM), interpolation=cv2.INTER_AREA)
    first2 = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
    if first2.shape[0] != IN_CAM:
        first2 = cv2.resize(first2, (IN_CAM, IN_CAM), interpolation=cv2.INTER_AREA)
    for _ in range(K):
        buf.append(first)
        buf2.append(first2)

    stage = "approach"
    frozen = first2.copy()
    mask_prev = False
    closed_t = lifted_t = placed_t = released_t = None
    dist_close = dist_close_xy = push_at_close = None
    push_max = 0.0
    zmax = z0
    min_dist = 1e9
    above_plate = False
    mask_steps = 0
    ema_alpha, prev_a = 0.8, None
    done = False
    step = 0
    for step in range(MAX_STEPS):
        # ---- 观测 ----
        g1 = float(np.asarray(obs["robot0_gripper_qpos"])[0])
        bowl = np.asarray(obs[target], dtype=float)
        z = float(bowl[2])
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        zmax = max(zmax, z)
        min_dist = min(min_dist, float(np.linalg.norm(eef - bowl)))
        push = float(np.linalg.norm((bowl - bowl0)[:2]))
        push_max = max(push_max, push)
        # ---- 阶段状态机 ----
        if stage == "approach" and g1 < G_CLOSED:
            stage = "grasp"
            closed_t = step
            dist_close = float(np.linalg.norm(eef - bowl))
            dist_close_xy = float(np.linalg.norm((eef - bowl)[:2]))
            push_at_close = push
        if stage == "grasp" and (z - z0) > LIFT_THR:
            stage = "carry"
            lifted_t = step
        if stage == "carry" and plate_xy is not None:
            if float(np.linalg.norm(bowl[:2] - plate_xy)) < PLATE_R:
                stage = "place"
                placed_t = step
                above_plate = True
        if released_t is None and closed_t is not None and g1 > G_OPEN:
            released_t = step
        # ---- 掩码：冻结或重放腕部视角 ----
        img2_full = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
        img2 = cv2.resize(img2_full, (IN_CAM, IN_CAM), interpolation=cv2.INTER_AREA) if img2_full.shape[0] != IN_CAM else img2_full
        if record_wrist is not None:
            record_wrist.append(img2.copy())
        mask_now = (mask_stage is not None) and (stage == mask_stage)
        if mask_now:
            mask_steps += 1
            if mask_mode == "replay" and replay_frames:
                wrist_used = replay_frames[min(step, len(replay_frames) - 1)]
            else:
                if not mask_prev:
                    frozen = img2.copy()          # 窗口开启瞬间冻结
                wrist_used = frozen
        else:
            wrist_used = img2
        mask_prev = mask_now
        # ---- 输入组装 ----
        img_full = image(obs)
        img = cv2.resize(img_full, (IN_CAM, IN_CAM), interpolation=cv2.INTER_AREA) if img_full.shape[0] != IN_CAM else img_full
        buf.append(img)
        buf2.append(wrist_used)
        stacked = np.concatenate(list(buf), axis=2)
        if n_cam == 2:
            stacked2 = np.concatenate(list(buf2), axis=2)
            stacked = np.concatenate([stacked, stacked2], axis=2)
        s = make_state(obs).astype(np.float32)
        x = torch.from_numpy(stacked.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        st = torch.from_numpy(s).float().unsqueeze(0).to(device)
        with torch.no_grad():
            a = model(x, st)[0].cpu().numpy()
        if prev_a is not None:
            a = ema_alpha * a + (1 - ema_alpha) * prev_a
        prev_a = a
        obs, reward, done, info = env.step(np.asarray(a, dtype=np.float32))
        if done:
            break

    bowl_end = np.asarray(obs[target], dtype=float)
    plate_dist_end = None
    if plate_xy is not None:
        plate_dist_end = float(np.linalg.norm(bowl_end[:2] - plate_xy))
    # ---- 失败标签 ----
    if done:
        label = None
    elif closed_t is None:
        label = "no_grasp"          # 从未闭爪（接近阶段失败）
    elif lifted_t is None:
        label = "grasp_fail"        # 闭爪但没抬起（滑脱/没夹住）
    elif not above_plate:
        label = "carry_fail"        # 抬起但从未进入盘区（搬运失败）
    else:
        label = "place_fail"        # 进入盘区但未放成功（放置失败/超时）
    return {
        "ep": idx, "success": bool(done), "steps": step + 1, "label": label,
        "closed_t": closed_t, "lifted_t": lifted_t, "placed_t": placed_t, "released_t": released_t,
        "dist_close": None if dist_close is None else round(dist_close, 4),
        "dist_close_xy": None if dist_close_xy is None else round(dist_close_xy, 4),
        "push_at_close": None if push_at_close is None else round(push_at_close, 4),
        "push_max": round(push_max, 4),
        "zmax_rel": round(zmax - z0, 4),
        "min_dist": round(min_dist, 4),
        "plate_dist_end": None if plate_dist_end is None else round(plate_dist_end, 4),
        "above_plate": bool(above_plate),
        "mask_steps": mask_steps,
    }


def run_batch(mode, tasks, dual, episodes, mask_stage=None, mask_mode="freeze"):
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    if os.path.isfile(RESULT_FILE):
        results = json.load(open(RESULT_FILE, encoding="utf-8"))
    print(f"[套件] {SUITE}  render={RENDER}  模式={mode}", flush=True)
    for tid in tasks:
        env, suite, task = build_env(tid, RENDER)
        h5 = os.path.join(LE.get_libero_path("datasets"), SUITE, task.language.replace(" ", "_") + "_demo.hdf5")
        mp = model_path_for(h5, dual)
        if not os.path.isfile(mp):
            print(f"[跳过] 模型不存在: {mp}", flush=True)
            env.close()
            continue
        model, n_cam = load_model(mp, device)
        inits = suite.get_task_init_states(tid)
        n = min(episodes, len(inits))
        tag = f"mask={mask_stage}" if mask_stage else "baseline"
        suite_tag = "" if SUITE == "libero_spatial" else SUITE.replace("libero_", "") + "_"
        key = f"{mode}|{'dual' if dual else 'single'}|{mask_stage}|{suite_tag}task{tid}"
        if mask_stage and mask_mode != "freeze":
            key = f"{mode}-{mask_mode}|{'dual' if dual else 'single'}|{mask_stage}|{suite_tag}task{tid}"
            tag += f"({mask_mode})"
        print(f"===== task{tid} {tag} ({'dual' if dual else 'single'}) {n} 局 =====", flush=True)
        t0 = time.time()
        replay_frames = None
        if mask_stage and mask_mode == "replay":
            rec = []
            print("  [录制] 先跑一局未掩码回合以录制真实腕部画面…", flush=True)
            run_episode(env, inits, 0, model, n_cam, device, record_wrist=rec, task_language=task.language)
            replay_frames = rec
            print(f"  [录制完成] {len(rec)} 帧", flush=True)
        eps = []
        succ = 0
        for i in range(n):
            r = run_episode(env, inits, i, model, n_cam, device, mask_stage=mask_stage,
                            mask_mode=mask_mode, replay_frames=replay_frames,
                            task_language=task.language)
            eps.append(r)
            succ += int(r["success"])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n}  累计 {succ}/{i+1}", flush=True)
        env.close()
        labels = {}
        for r in eps:
            lab = r["label"] or "success"
            labels[lab] = labels.get(lab, 0) + 1
        results[key] = {
            "task": tid, "mode": mode, "dual": dual, "mask_stage": mask_stage,
            "succ": succ, "n": n, "rate": round(succ / n * 100, 1),
            "labels": labels, "episodes": eps, "seconds": round(time.time() - t0, 1),
        }
        json.dump(results, open(RESULT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[完成] {key}: {succ}/{n} = {succ/n*100:.1f}%  失败分布={ {k:v for k,v in labels.items() if k!='success'} }"
              f"  耗时 {time.time()-t0:.0f}s", flush=True)


def main():
    global SUITE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "mask"])
    ap.add_argument("--tasks", default="2,9,7,8")
    ap.add_argument("--stage", default="grasp", choices=["approach", "grasp", "carry", "place"])
    ap.add_argument("--mask-mode", default="freeze", choices=["freeze", "replay"])
    ap.add_argument("--dual", action="store_true")
    ap.add_argument("--eps", type=int, default=EPISODES)
    ap.add_argument("--suite", default=None,
                    help="套件：libero_spatial（默认）/ libero_object / …")
    args = ap.parse_args()
    if args.suite:
        SUITE = args.suite
    tasks = [int(x) for x in args.tasks.split(",") if x.strip()]
    if args.mode == "baseline":
        run_batch("baseline", tasks, dual=args.dual, episodes=args.eps)
    else:
        run_batch("mask", tasks, dual=True, episodes=args.eps, mask_stage=args.stage,
                  mask_mode=args.mask_mode)


if __name__ == "__main__":
    main()
