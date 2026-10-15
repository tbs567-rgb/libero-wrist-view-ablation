# -*- coding: utf-8 -*-
"""
官方 LIBERO 数据集版：人类遥操作演示（HDF5）训练帧堆叠 BC，50 初始态标准评测。

数据来源：LIBERO 官方 HuggingFace 数据集（人类遥操作演示，非规则专家），
协议 = 官方标准：50 条演示全用于训练，在任务全部 50 个初始状态上评测成功率。
这才是能与 LIBERO leaderboard 对比的协议。
"""
import os
import sys
import argparse
import glob
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
VENV = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if os.environ.get("_LIBERO_REEXEC") != "1" and os.path.isfile(VENV):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV):
        import subprocess
        env = dict(os.environ, _LIBERO_REEXEC="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        raise SystemExit(subprocess.call([VENV, os.path.abspath(__file__)] + sys.argv[1:], env=env))

import libero_env as LE  # noqa: E402,F401  (mujoco 必须先于 torch)
import numpy as np  # noqa: E402
from low_resource_bc import (  # noqa: E402
    make_env, image, make_state, EVAL_CAM, TRAIN_INITS, TASK_ID, MAX_STEPS, MODEL_DIR, CAM,
)
from low_resource_framestack import FrameStackBC, K  # noqa: E402
from low_resource_resnet_fs import ResNetFSBC  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

DATASET_DIR = os.path.join(HERE, "datasets", "libero_spatial")


class WideFrameStackBC(nn.Module):
    """加宽版帧堆叠网络（~115 万参数）：官方数据容量需求比规则专家数据大。"""

    def __init__(self, k=K, state_dim=8):
        super().__init__()
        self.k = k
        self.vision = nn.Sequential(
            nn.Conv2d(3 * k, 32, 5, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),      # 128*16=2048
        )
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2048 + 64, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 7), nn.Tanh(),
        )

    def forward(self, x, state):
        return self.head(torch.cat([self.vision(x), self.state(state)], dim=1))


def load_official_hdf5(h5_path, cam=96, k=K, dual=False, wrist=False):
    """读官方 LIBERO HDF5：50 条人类演示 → 帧堆叠训练集。

    dual=True 时同时使用外部视角（agentview）与腕部相机（eye_in_hand），
    通道排列 = [agentview K 帧(3K), eye_in_hand K 帧(3K)]。
    wrist=True 时仅用腕部相机（单相机结构 n_cam=1）——三组对照（仅全局/仅腕部/双视角）的第三组。
    proprio = ee_pos(3) + ee_ori(3, axis-angle) + gripper_states(2)，与评测时 make_state 同源。
    """
    f = h5py.File(h5_path, "r")
    demos = sorted(f["data"].keys())
    imgs, props, acts = [], [], []
    for dn in demos:
        d = f["data"][dn]
        src = "obs/eye_in_hand_rgb" if wrist else "obs/agentview_rgb"
        rgb = d[src][()][:, ::-1]                          # mujoco 图上下颠倒 → 翻正
        rgb = np.stack([cv2.resize(f2, (cam, cam), interpolation=cv2.INTER_AREA) for f2 in rgb])
        rgb2 = None
        if dual:
            rgb2 = d["obs/eye_in_hand_rgb"][()][:, ::-1]
            rgb2 = np.stack([cv2.resize(f2, (cam, cam), interpolation=cv2.INTER_AREA) for f2 in rgb2])
        T = len(rgb)
        ee_pos = d["obs/ee_pos"][()]
        ee_ori = d["obs/ee_ori"][()]
        grip = d["obs/gripper_states"][()]
        proprio = np.concatenate((ee_pos, ee_ori, grip), axis=1).astype(np.float32)
        acts.append(d["actions"][()][k - 1:].astype(np.float32))   # 对齐帧窗口（丢前 k-1 步）
        for t in range(k - 1, T):
            stacked = np.stack([rgb[t - k + j] for j in range(k)])          # (k, cam, cam, 3)
            stacked = stacked.transpose(3, 0, 1, 2).reshape(3 * k, cam, cam)  # (3k, cam, cam)
            if dual:
                stacked2 = np.stack([rgb2[t - k + j] for j in range(k)])
                stacked2 = stacked2.transpose(3, 0, 1, 2).reshape(3 * k, cam, cam)
                stacked = np.concatenate([stacked, stacked2], axis=0)         # (6k, cam, cam)
            imgs.append(stacked)
            props.append(proprio[t])
    f.close()
    actions = torch.from_numpy(np.concatenate(acts, axis=0)).float()
    images = torch.from_numpy(np.stack(imgs)).float() / 255.0
    proprio = torch.from_numpy(np.stack(props)).float()
    return images, proprio, actions


def train(h5_path, epochs=600, batch=64, dual=False, seed=7, wrist=False):
    from torch.utils.data import TensorDataset, DataLoader, random_split
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"[种子] {seed}", flush=True)
    print(f"读取官方演示: {os.path.basename(h5_path)}  dual_cam={dual}  wrist={wrist}")
    images, proprio, actions = load_official_hdf5(h5_path, cam=96, dual=dual, wrist=wrist)
    print(f"帧堆叠后样本数={len(actions)}（K={K} 帧堆叠，图像 {images.shape[2]}x{images.shape[3]}）")
    ds = TensorDataset(images, proprio, actions)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    vloader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFSBC(state_dim=8, n_cam=2 if dual else 1).to(device)
    # 预训练骨干温和微调（上次 0% 的教训：40 epoch 恒定大 lr 对 ResNet 太猛）
    bb_params = [p for n, p in model.named_parameters() if "vision" in n]
    head_params = [p for n, p in model.named_parameters() if "vision" not in n]
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": 5e-5},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"[官方数据 ResNet{' 双相机' if dual else ''}] 设备={device} 参数量={sum(p.numel() for p in model.parameters()):,} "
          f"训练={len(train_ds)} 验证={len(val_ds)} epochs={epochs}")
    tag = "_wrist" if wrist else ("_dual" if dual else "")
    seed_tag = "" if seed == 7 else f"_s{seed}"
    out_path = os.path.join(MODEL_DIR, f"official_resnet{tag}{seed_tag}_" + os.path.basename(h5_path).replace("_demo.hdf5", ".pt"))
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0
        for x, s, y in loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            # —— 数据增强（对 12 通道帧堆叠图逐帧同步做）——
            if torch.rand(1).item() < 0.5:                      # 随机平移 ±3px（wrap）
                dx, dy = int(torch.randint(-3, 4, (1,))), int(torch.randint(-3, 4, (1,)))
                x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
            x = x * torch.empty(1, 1, 1, 1, device=x.device).uniform_(0.9, 1.1)   # 亮度抖动
            x = x + torch.randn_like(x) * 0.02                                    # 高斯噪声
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(x, s), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(x)
        sched.step()
        if ep % 25 == 0 or ep == 1 or ep == epochs:
            model.eval(); val = 0.0
            with torch.no_grad():
                for x, s, y in vloader:
                    val += loss_fn(model(x.to(device), s.to(device)), y.to(device)).item() * len(x)
            val /= max(1, len(val_ds))
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "cam": CAM, "task": TASK_ID,
                            "version": "official_fs", "K": K, "state_dim": 8, "n_cam": 2 if dual else 1}, out_path)
            print(f"  epoch {ep:03d}/{epochs}  train_mse={total/len(train_ds):.6f}  val_mse={val:.6f}", flush=True)
    torch.save({"model": model.state_dict(), "cam": CAM, "task": TASK_ID,
                "version": "official_fs", "K": K, "state_dim": 8, "n_cam": 2 if dual else 1}, out_path)
    print(f"已保存: {out_path}（best val_mse={best_val:.6f}）")
    return out_path


def evaluate_all_inits(model_path, episodes=50, save_video=False, task_id=TASK_ID, in_cam=96,
                       suite_name="libero_spatial", wrist=False):
    """官方标准协议：在任务全部 50 个初始状态上评测（不分 seen/unseen）。
    in_cam 必须与训练时 load_official_hdf5 的 cam 一致（ResNet/宽网络版=96）。
    suite_name 支持跨套件评测（如 libero_object）——env 按 (suite, task) 构建。"""
    import cv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    n_cam = int(ckpt.get("n_cam", 1))
    model = ResNetFSBC(state_dim=8, n_cam=n_cam).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    suite = LE.get_suite(suite_name)
    task = suite.get_task(task_id)
    bddl = os.path.join(LE.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = LE.OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"],
                                camera_heights=EVAL_CAM, camera_widths=EVAL_CAM)
    env.seed(0)
    inits = suite.get_task_init_states(task_id)
    n = min(episodes, len(inits))
    print(f"官方协议评测: {task.language}; 初始态 0..{n-1} 共 {n} 局（动作 EMA 平滑）")
    succ = 0
    ema_alpha = 0.8
    for i in range(n):
        env.reset()
        obs = env.set_init_state(inits[i])
        buf = deque(maxlen=K)
        buf2 = deque(maxlen=K)
        if wrist:
            first = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
        else:
            first = image(obs)
        if first.shape[0] != in_cam:
            first = cv2.resize(first, (in_cam, in_cam), interpolation=cv2.INTER_AREA)
        first2 = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
        if first2.shape[0] != in_cam:
            first2 = cv2.resize(first2, (in_cam, in_cam), interpolation=cv2.INTER_AREA)
        for _ in range(K):
            buf.append(first)
            buf2.append(first2)
        done = False
        prev_a = None
        for step in range(MAX_STEPS):
            if wrist:
                img_full = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
            else:
                img_full = image(obs)
            img = cv2.resize(img_full, (in_cam, in_cam), interpolation=cv2.INTER_AREA) if img_full.shape[0] != in_cam else img_full
            buf.append(img)
            if n_cam == 2:
                img2_full = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].astype(np.uint8)
                img2 = cv2.resize(img2_full, (in_cam, in_cam), interpolation=cv2.INTER_AREA) if img2_full.shape[0] != in_cam else img2_full
                buf2.append(img2)
            stacked = np.concatenate(list(buf), axis=2)
            if n_cam == 2:
                stacked2 = np.concatenate(list(buf2), axis=2)
                stacked = np.concatenate([stacked, stacked2], axis=2)
            s = make_state(obs).astype(np.float32)
            x = torch.from_numpy(stacked.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
            st = torch.from_numpy(s).float().unsqueeze(0).to(device)
            with torch.no_grad():
                a = model(x, st)[0].cpu().numpy()
            if prev_a is not None:                      # 指数滑动平均：抑制逐帧抖动
                a = ema_alpha * a + (1 - ema_alpha) * prev_a
            prev_a = a
            obs, reward, done, info = env.step(np.asarray(a, dtype=np.float32))
            if done:
                break
        succ += int(done)
        if (i + 1) % 10 == 0 or done:
            print(f"  init={i:02d}  steps={step+1:3d}  success={done}  累计 {succ}/{i+1}", flush=True)
    env.close()
    print(f"官方协议成功率: {succ}/{n} = {succ/n*100:.1f}%")
    return succ, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "all"])
    ap.add_argument("--h5", default=None, help="官方 HDF5 路径（默认 task0）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--task-id", type=int, default=TASK_ID, help="评测哪个任务（多任务 benchmark 必须传对）")
    ap.add_argument("--suite", default="libero_spatial", help="所属套件（跨套件抽检用 libero_object）")
    ap.add_argument("--dual-cam", action="store_true", help="使用双相机（agentview + eye_in_hand）")
    ap.add_argument("--wrist-cam", action="store_true", help="仅使用腕部相机（三组对照的第三组，单相机结构）")
    ap.add_argument("--seed", type=int, default=7, help="训练/评测种子（非 7 时模型文件带 _s<seed> 后缀）")
    args = ap.parse_args()
    h5 = args.h5 or glob.glob(os.path.join(DATASET_DIR, "*between_the_plate_and_the_ramekin*"))[0]
    if args.mode in ("train", "all"):
        model_path = train(h5, args.epochs, dual=args.dual_cam, seed=args.seed, wrist=args.wrist_cam)
    if args.mode in ("eval", "all"):
        if args.mode == "all":
            mp = model_path
        elif args.h5:
            tag = "_wrist" if args.wrist_cam else ("_dual" if args.dual_cam else "")
            seed_tag = "" if args.seed == 7 else f"_s{args.seed}"
            mp = os.path.join(MODEL_DIR, f"official_resnet{tag}{seed_tag}_" + os.path.basename(args.h5).replace("_demo.hdf5", ".pt"))
        else:
            mp = sorted(glob.glob(os.path.join(MODEL_DIR, "official_resnet_*")))[-1]
        evaluate_all_inits(mp, args.episodes, task_id=args.task_id, suite_name=args.suite, wrist=args.wrist_cam)


if __name__ == "__main__":
    main()
