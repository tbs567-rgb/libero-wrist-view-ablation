# -*- coding: utf-8 -*-
"""
低显存 LIBERO 实验 v2：带目标识别的视觉行为克隆

题目：基于 LIBERO 的低显存视觉—动作策略训练与资源受限评测
流程：
  collect  由规则专家在 LIBERO-Spatial task 0 采集演示
  train    图像 CNN + (本体感觉 8 维 + 碗识别特征 4 维) → 7 维动作
  eval     双组评测：训练见过的初始态 vs 从未见过的初始态（位姿泛化分数）

v2 升级点（相对 v1）：
  1. find_bowl()：从 64x64 图像里用暗像素分割 + 连通域识别黑碗，
     输出 [质心x, 质心y, 尺寸, 是否检出] 4 维特征 —— 模型"先看清碗再动手"
  2. 识别特征与本体感觉一起过 MLP 再与视觉特征拼接，策略可随碗的位置调整
  3. eval 分 seen / unseen 两组初始态，成功率差值 = 位姿泛化分数
  4. 旧 npz 数据结构不变（碗特征训练时从图像现算），旧数据可直接复用
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
VENV = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if os.environ.get("_LIBERO_REEXEC") != "1" and os.path.isfile(VENV):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV):
        import subprocess
        env = dict(os.environ, _LIBERO_REEXEC="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        raise SystemExit(subprocess.call([VENV, os.path.abspath(__file__)] + sys.argv[1:], env=env))

import libero_env as LE  # noqa: E402
import numpy as np  # noqa: E402

DATA_DIR = os.path.join(HERE, "low_resource_data")
MODEL_DIR = os.path.join(HERE, "low_resource_models")
DATA_FILE = os.path.join(DATA_DIR, "spatial_task0_bc.npz")
MODEL_FILE = os.path.join(MODEL_DIR, "cnn_bc_task0_v2.pt")
FINAL_FILE = os.path.join(MODEL_DIR, "cnn_bc_task0_v2_FINAL.pt") # 任务成功率最高的模型（人工选定封存）
BEST_FILE = os.path.join(MODEL_DIR, "cnn_bc_task0_v2_best.pt")   # 验证误差最低的那一轮自动存这里
SUITE_NAME = "libero_spatial"
TASK_ID = 0
CAM = 64
EVAL_CAM = int(os.environ.get("LIBERO_EVAL_CAM", "64"))   # 评测渲染分辨率：默认 64 与训练一致；高清录像设 256
MAX_STEPS = 300
N_INIT_STATES = 50          # LIBERO 每个任务预置 50 个初始状态
TRAIN_INITS = 40            # 采集/训练只用前 40 个，后 10 个留给 unseen 评测


def make_env(cam=CAM):
    suite = LE.get_suite(SUITE_NAME)
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(LE.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = LE.OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"], camera_heights=cam, camera_widths=cam)
    env.seed(0)
    return env, suite, task


def find_target(obs):
    keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot") and "_to_robot" not in k]
    return keys[0] if keys else None


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def make_state(obs):
    """8 维 proprio：eef xyz + axis-angle + gripper qpos。"""
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])).astype(np.float32)


def image(obs):
    # MuJoCo 图像上下颠倒；训练统一成 RGB、[0,1]、64x64
    return np.asarray(obs["agentview_image"])[::-1].astype(np.uint8)


# 黑碗 1 在初始态 0 的标定像素位置（用相机投影标定一次；固定任务里碗只会移动 ~2px）
BOWL_PRIOR = (0.75, 0.56)


def find_bowl(img):
    """从 64x64 RGB 图像里识别任务目标黑碗。

    场景实测（相机投影 + 连通域对号）：黑碗在此光照下呈中性灰（RGB~82），
    深褐盒子更暗（RGB~33）会被亮度下限排除，机械臂是巨大灰块会被面积上限排除，
    白盘子不进暗色掩码，曲奇带褐色会被低饱和要求排除。
    三重过滤：中性灰 + 面积适中 + 离场景先验位置最近。

    返回 4 维特征 [cx, cy, size, detected]，全部归一化：
      cx, cy   碗的归一化坐标；size = sqrt(面积)/图像高；detected 1/0
    纯 numpy、无状态、可复现；训练和评测用同一个函数。
    """
    a = np.asarray(img).astype(np.int16)
    mx = a.max(axis=2)
    mask = (mx >= 45) & (mx <= 130) & ((mx - a.min(axis=2)) <= 25)   # 中性灰
    mask = mask.astype(bool)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_d, best_feat = 1e9, None
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            n = len(comp)
            if n < 12 or n > 250:
                continue
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            cx, cy = float(np.mean(xs)) / W, float(np.mean(ys)) / H
            d = ((cx - BOWL_PRIOR[0]) ** 2 + (cy - BOWL_PRIOR[1]) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_feat = (cx, cy, min(float(np.sqrt(n)) / float(H), 1.0))
    if best_feat is None or best_d > 0.18:
        return np.array([BOWL_PRIOR[0], BOWL_PRIOR[1], 0.0, 0.0], dtype=np.float32)
    return np.array([best_feat[0], best_feat[1], best_feat[2], 1.0], dtype=np.float32)


def make_input_state(obs):
    """模型完整输入：8 维 proprio + 4 维碗识别特征 = 12 维。"""
    return np.concatenate((make_state(obs), find_bowl(image(obs)))).astype(np.float32)


class ExpertPolicy:
    """六阶段状态机专家（用仿真器特权位置，只用于造演示数据）。

    与"按步数切阶段"不同，这里按【状态是否到达】推进阶段：
    同一个状态永远对应同一个动作，行为克隆学到的才是纯状态反馈，不会自相矛盾。
    阶段：悬停 → 下降 → 闭爪 → 抬起 → 平移到盘子上方 → 降下松爪放碗
    """

    GAIN = 4.0

    def __init__(self):
        self.phase = 0
        self.t_in = 0
        self.lift_z = 0.0
        self.plate_z = 0.0

    def reset(self):
        self.phase, self.t_in = 0, 0

    def _move(self, eef, tgt_xy, tgt_z):
        a = np.zeros(7, dtype=np.float32)
        a[:2] = np.clip((np.asarray(tgt_xy) - eef[:2]) * self.GAIN, -1, 1)
        a[2] = np.clip((tgt_z - eef[2]) * self.GAIN, -1, 1)
        return a

    def act(self, obs, target):
        self.t_in += 1
        a = np.zeros(7, dtype=np.float32)
        if not (target and target in obs):
            return a
        eef = np.asarray(obs["robot0_eef_pos"])
        obj = np.asarray(obs[target])
        plate = np.asarray(obs.get("plate_1_pos", obj))
        if self.phase == 0:      # 悬停在碗上方 0.10，等 xy、z 都到位
            a = self._move(eef, obj[:2], obj[2] + 0.10)
            a[6] = 1.0
            if self.t_in >= 10 and np.linalg.norm(obj[:2] - eef[:2]) < 0.012 and abs(obj[2] + 0.10 - eef[2]) < 0.015:
                self.phase, self.t_in = 1, 0
        elif self.phase == 1:    # 缓慢下降到碗口（手指先碰到碗沿即算到位）
            a = self._move(eef, obj[:2], obj[2] + 0.02)
            a[6] = 1.0
            if self.t_in >= 12 and abs(obj[2] + 0.02 - eef[2]) < 0.015:
                self.phase, self.t_in = 2, 0
        elif self.phase == 2:    # 闭爪夹住（给足闭合时间）
            a[6] = -1.0
            if self.t_in >= 35:
                self.phase, self.t_in = 3, 0
                self.lift_z = float(obj[2]) + 0.12
        elif self.phase == 3:    # 抬起 0.12
            a = self._move(eef, eef[:2], self.lift_z)
            a[6] = -1.0
            if self.t_in >= 10 and eef[2] >= self.lift_z - 0.012:
                self.phase, self.t_in = 4, 0
                self.plate_z = float(plate[2])
        elif self.phase == 4:    # 平移到盘子正上方（保持高度）
            a = self._move(eef, plate[:2], self.plate_z + 0.12)
            a[6] = -1.0
            if self.t_in >= 10 and np.linalg.norm(plate[:2] - eef[:2]) < 0.012 and abs(self.plate_z + 0.12 - eef[2]) < 0.015:
                self.phase, self.t_in = 5, 0
        else:                    # 缓降 + 松爪，把碗放到盘子上
            a = self._move(eef, plate[:2], self.plate_z + 0.035)
            a[6] = 1.0
        return a


def expert_action(obs, target, step):
    """兼容旧接口的按步数专家（不推荐用于采集，仅调试用）。"""
    ep = ExpertPolicy()
    return ep.act(obs, target)


def collect(episodes=40, noise=0.0, dirty_label=False):
    """采集演示数据。LIBERO_RENDER_CAM=256 可用高清渲染采集（存图前缩回 64，
    训练输入分布与 256 高清评测一致）。

    noise>0 时开启域随机化：专家平移动作每步加高斯噪声再执行。
    dirty_label=False（默认）：标签=干净专家动作（教科书式 noise-injection DAgger）。
    dirty_label=True：标签=实际执行的加噪动作（等效标签平滑，实测更稳）。
    每回合随机亮度/对比度扰动（±8%）治渲染敏感。
    """
    import cv2
    render = int(os.environ.get("LIBERO_RENDER_CAM", str(CAM)))
    os.makedirs(DATA_DIR, exist_ok=True)
    xs_img, xs_state, ys = [], [], []
    env, suite, task = make_env(render)
    inits = suite.get_task_init_states(TASK_ID)
    print(f"采集任务: {task.language}")
    print(f"渲染={render} 存图={CAM}  episodes={episodes}（初始态 0..{TRAIN_INITS-1}）  动作噪声σ={noise}")
    successes = 0
    for ep in range(episodes):
        env.reset()
        obs = env.set_init_state(inits[ep % TRAIN_INITS])
        target = find_target(obs)
        expert = ExpertPolicy()
        rng = np.random.default_rng(1000 + ep)          # 每回合独立、可复现
        bri = rng.uniform(0.92, 1.08)                   # 亮度
        con = rng.uniform(0.92, 1.08)                   # 对比度
        for step in range(MAX_STEPS):
            img_full = image(obs)
            img = cv2.resize(img_full, (CAM, CAM), interpolation=cv2.INTER_AREA) if img_full.shape[0] != CAM else img_full
            img_aug = np.clip((img.astype(np.float32) - 128.0) * con * bri + 128.0, 0, 255).astype(np.uint8)
            xs_img.append(img_aug)
            xs_state.append(make_state(obs))
            action = expert.act(obs, target)
            a_exec = action
            if noise > 0:                            # 执行=加噪动作，把状态推进到偏离处
                a_exec = action.copy()
                a_exec[:3] = np.clip(a_exec[:3] + rng.normal(0, noise, 3), -1, 1)
            ys.append(a_exec if dirty_label else action)   # 标签可选干净或带噪（标签平滑）
            obs, reward, done, info = env.step(a_exec)
            if done:
                successes += 1
                break
        print(f"  episode {ep+1:02d}/{episodes}  steps={step+1:3d}  done={done}", flush=True)
    env.close()
    np.savez_compressed(DATA_FILE, images=np.asarray(xs_img), states=np.asarray(xs_state), actions=np.asarray(ys))
    print(f"数据已保存: {DATA_FILE}")
    print(f"样本数={len(ys)}, expert_done={successes}/{episodes}")


def bowl_features(images_np):
    """批量从图像算碗识别特征（训练时一次性预处理）。"""
    return np.stack([find_bowl(im) for im in images_np]).astype(np.float32)


def build_model():
    import torch
    import torch.nn as nn
    class CNNBC(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Sequential(
                nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
                nn.Conv2d(16, 32, 5, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            )
            # v2: 12 维输入（8 proprio + 4 碗识别）先过一个小 MLP
            self.state = nn.Sequential(nn.Linear(12, 64), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(64 * 4 * 4 + 64, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 7), nn.Tanh(),
            )
        def forward(self, x, state):
            return self.head(torch.cat([self.vision(x), self.state(state)], dim=1))
    return CNNBC


def train(epochs=40, batch=64):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split
    torch.manual_seed(7)          # 训练可复现（否则每次结果都不同，没法对比实验）
    np.random.seed(7)
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"没有数据，请先运行 collect：{DATA_FILE}")
    d = np.load(DATA_FILE)
    images_np = d["images"]
    states = d["states"]
    actions = d["actions"]
    print(f"加载 {len(actions)} 个样本，正在从图像现算碗识别特征……")
    t0 = time.time()
    bowl = bowl_features(images_np)
    det_rate = float(bowl[:, 3].mean())
    print(f"识别完成，用时 {time.time()-t0:.1f}s；检出率={det_rate*100:.1f}%，质心均值=({bowl[:,0].mean():.2f}, {bowl[:,1].mean():.2f})")
    inputs = np.concatenate((states, bowl), axis=1)          # 8 + 4 = 12 维
    images = torch.from_numpy(images_np.transpose(0, 3, 1, 2)).float() / 255.0
    inputs = torch.from_numpy(inputs).float()
    actions = torch.from_numpy(actions).float()
    ds = TensorDataset(images, inputs, actions)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(7))
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    vloader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNNBC = build_model()
    model = CNNBC().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"训练设备: {device}; torch={torch.__version__}; 参数量={sum(p.numel() for p in model.parameters()):,}")
    print(f"训练样本={len(train_ds)}, 验证样本={len(val_ds)}, batch={batch}, epochs={epochs}")
    best_val, best_ep = float("inf"), 0
    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for x, s, y in loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x, s); loss = loss_fn(pred, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(x)
        model.eval(); val=0.0
        with torch.no_grad():
            for x, s, y in vloader:
                val += loss_fn(model(x.to(device), s.to(device)), y.to(device)).item() * len(x)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"epoch {ep:02d}/{epochs}  train_mse={total/len(train_ds):.6f}  val_mse={val/len(val_ds):.6f}", flush=True)
        val_mean = val / len(val_ds)
        if val_mean < best_val:   # 验证集误差创新低 → 立刻存一份 best
            best_val, best_ep = val_mean, ep
            torch.save({"model": model.state_dict(), "cam": CAM, "task": TASK_ID, "version": 2,
                        "best_epoch": ep, "best_val_mse": best_val}, BEST_FILE)
            print(f"  ** epoch {ep} 验证误差新低 {val_mean:.6f}，已保存 best", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "cam": CAM, "task": TASK_ID, "version": 2}, MODEL_FILE)
    print(f"最终模型已保存: {MODEL_FILE}")
    print(f"全程最佳: epoch {best_ep}（val_mse={best_val:.6f}）→ {BEST_FILE}")


def run_policy_episode(env, inits, init_idx, model, device, writer=None, viewer=None, slow=0.03, log_prefix="", cam_in=CAM):
    """跑一个回合的已训练策略，返回 (success, steps)。

    env 可以是高分辨率渲染（EVAL_CAM=256）：喂模型/find_bowl 前先缩回 cam_in=64，
    录像写原始高清帧，所以画质和模型输入互不影响。
    """
    import torch
    import cv2
    env.reset()
    obs = env.set_init_state(inits[init_idx])
    done = False
    det = 0
    for step in range(MAX_STEPS):
        img_full = image(obs)
        if img_full.shape[0] != cam_in:
            img = cv2.resize(img_full, (cam_in, cam_in), interpolation=cv2.INTER_AREA)
        else:
            img = img_full
        s = np.concatenate((make_state(obs), find_bowl(img))).astype(np.float32)
        det += int(s[11] > 0.5)
        x = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        st = torch.from_numpy(s).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(x, st)[0].cpu().numpy()
        obs, reward, done, info = env.step(action)
        if writer is not None:
            writer.write(image(obs)[:, :, ::-1])   # 已翻正 RGB -> BGR（高清帧）
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                done = True
                break
            time.sleep(slow)
        if done:
            break
    rate = det / max(1, step + 1)
    print(f"  {log_prefix} init={init_idx:02d}  steps={step+1:3d}  碗检出率={rate*100:0.0f}%  success={done}", flush=True)
    return bool(done), step + 1


def evaluate(episodes=5, show=False, save_video=True, slow=0.03, model_path=None):
    """双组评测：seen（训练用过的初始态）vs unseen（从未见过的初始态）。

    成功率差值 = 位姿泛化分数（越小泛化越好）。
    """
    import torch
    CNNBC = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBC().to(device)
    load_path = model_path or next(
        f for f in (FINAL_FILE, BEST_FILE, MODEL_FILE) if os.path.isfile(f)
    )
    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    extra = f"（epoch {ckpt['best_epoch']}，val_mse={ckpt['best_val_mse']:.6f}）" if "best_epoch" in ckpt else ""
    print(f"已加载模型: {load_path}{extra}")
    env, suite, task = make_env(EVAL_CAM)
    inits = suite.get_task_init_states(TASK_ID)
    viewer = None
    video_dir = os.path.join(HERE, "output")
    if show:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(env.sim.model._model, env.sim.data._data)
        print("三维窗口已打开：可以拖动视角，关闭窗口会结束当前回合。")
    print(f"评测任务: {task.language}; device={device}")

    def run_group(tag, idx0, n):
        import cv2
        successes = 0
        files = []
        for i in range(n):
            tmp_path = os.path.join(video_dir, "_tmp_ep.mp4")
            writer = None
            if save_video:
                os.makedirs(video_dir, exist_ok=True)
                writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (EVAL_CAM, EVAL_CAM))
            ok, _ = run_policy_episode(env, inits, idx0 + i, model, device, writer, viewer, slow, log_prefix=f"[{tag}]")
            if writer is not None:
                writer.release()
                if ok:   # 成功的回合单独存档
                    dst = os.path.join(video_dir, f"success_{tag}_init{idx0+i:02d}.mp4")
                    os.replace(tmp_path, dst)
                    files.append(dst)
                else:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            successes += int(ok)
            if viewer is not None and not viewer.is_running():
                break
        return successes, files

    print(f"— seen 组：初始态 0..{episodes-1}（训练时见过）")
    seen, seen_files = run_group("seen", 0, episodes)
    unseen, unseen_files = 0, []
    if viewer is None or viewer.is_running():
        print(f"— unseen 组：初始态 {TRAIN_INITS}..{TRAIN_INITS+episodes-1}（训练从未见过）")
        unseen, unseen_files = run_group("unseen", TRAIN_INITS, episodes)
    if viewer is not None:
        viewer.close()
    env.close()
    print(f"seen    成功率: {seen}/{episodes} = {seen/episodes*100:.1f}%")
    print(f"unseen  成功率: {unseen}/{episodes} = {unseen/episodes*100:.1f}%")
    print(f"位姿泛化分数（seen - unseen）: {(seen-unseen)/episodes*100:+.1f} 个百分点")
    if seen_files or unseen_files:
        print("成功回合录像：")
        for p in seen_files + unseen_files:
            print("  " + p)


def debug_vision(out_dir=None):
    """识别模块自检：跑 1 回合，存 3 帧标注 PNG（红=识别位置，蓝=投影真值）。"""
    from PIL import Image, ImageDraw
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    if out_dir is None:
        out_dir = os.path.join(HERE, "output", "vision_debug")
    os.makedirs(out_dir, exist_ok=True)
    env, suite, task = make_env()
    inits = suite.get_task_init_states(TASK_ID)
    w2p = get_camera_transform_matrix(env.sim, "agentview", CAM, CAM)
    env.reset()
    obs = env.set_init_state(inits[0])
    target = find_target(obs)
    expert = ExpertPolicy()
    wanted = {0: "初始视角", 70: "下降阶段", 120: "抓取阶段", 170: "放置阶段"}
    saved = []
    for step in range(MAX_STEPS):
        if step in wanted:
            img = image(obs)
            feat = find_bowl(img)
            gt = project_points_from_world_to_camera(
                np.asarray(obs[target])[None, :], w2p, CAM, CAM)[0]     # [row, col]
            gx, gy = float(gt[1]) / CAM, float(gt[0]) / CAM
            err = ((feat[0] - gx) ** 2 + (feat[1] - gy) ** 2) ** 0.5 * CAM
            pil = Image.fromarray(img)
            dr = ImageDraw.Draw(pil)
            cx, cy = int(feat[0] * CAM), int(feat[1] * CAM)
            color = (255, 0, 0) if feat[3] > 0.5 else (0, 0, 255)
            dr.line((cx - 8, cy, cx + 8, cy), fill=color, width=2)
            dr.line((cx, cy - 8, cx, cy + 8), fill=color, width=2)
            dr.rectangle((cx - 10, cy - 10, cx + 10, cy + 10), outline=color, width=1)
            dr.ellipse((gx * CAM - 6, gy * CAM - 6, gx * CAM + 6, gy * CAM + 6), outline=(0, 96, 255), width=2)
            p = os.path.join(out_dir, f"frame_{step:03d}_{wanted[step]}.png")
            pil.resize((256, 256), Image.NEAREST).save(p)
            saved.append(p)
            print(f"step {step:3d} [{wanted[step]}] 识别: cx={feat[0]:.2f} cy={feat[1]:.2f} size={feat[2]:.2f} 检出={feat[3]:.0f}"
                  f" | 真值: ({gx:.2f}, {gy:.2f}) | 误差 {err:.1f}px")
        action = expert.act(obs, target)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    print(f"标注图已保存到: {out_dir}")
    for p in saved:
        print("  " + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "train", "eval", "all", "debug-vision"])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--show", action="store_true", help="评测时打开 MuJoCo 三维窗口")
    ap.add_argument("--no-video", action="store_true", help="不保存评测 MP4")
    ap.add_argument("--model", default=None, help="指定要评测的模型文件（默认用 best/最新）")
    ap.add_argument("--noise", type=float, default=0.0, help="采集时专家动作噪声σ（域随机化）")
    ap.add_argument("--dirty-label", action="store_true", help="标签用实际执行的加噪动作（标签平滑，实测更稳）")
    args = ap.parse_args()
    if args.mode == "debug-vision":
        debug_vision()
    if args.mode in ("collect", "all"):
        collect(args.episodes, args.noise, args.dirty_label)
    if args.mode in ("train", "all"):
        train(args.epochs)
    if args.mode in ("eval", "all"):
        evaluate(args.episodes, show=args.show, save_video=not args.no_video, model_path=args.model)


if __name__ == "__main__":
    main()
