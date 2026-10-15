# -*- coding: utf-8 -*-
"""
最强组合：ResNet18 预训练视觉骨干（ImageNet 权重平滑迁移到 12 通道）+ 帧堆叠时序。
预训练先验 + 运动感知双管齐下，6GB 显存可训。
"""
import os
import sys
import argparse
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

import libero_env as LE  # noqa: E402,F401
import numpy as np  # noqa: E402
from low_resource_bc import (  # noqa: E402
    make_env, image, make_state, find_bowl, find_target,
    DATA_FILE, MODEL_DIR, CAM, EVAL_CAM, TRAIN_INITS, TASK_ID, MAX_STEPS,
)
from low_resource_framestack import load_fs_dataset, K  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

RESNET_FS_MODEL = os.path.join(MODEL_DIR, "cnn_bc_task0_resnet_fs.pt")


class ResNetFSBC(nn.Module):
    """ResNet18 预训练骨干（conv1 扩到 3*K=12 通道，权重按 repeat/K 平滑迁移）+ 帧堆叠时序。"""

    def __init__(self, k=K, state_dim=12, n_cam=1):
        super().__init__()
        self.k = k
        self.n_cam = n_cam
        import torchvision.models as tvm
        bb = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # 第一层 conv 从 3 通道扩到 3*k*n_cam 通道（帧数 × 相机数），预训练权重平滑复制
        groups = k * n_cam
        old_w = bb.conv1.weight.data                      # (64,3,7,7)
        new_conv1 = nn.Conv2d(3 * groups, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(old_w.repeat(1, groups, 1, 1) / groups)
        bb.conv1 = new_conv1
        self.vision = nn.Sequential(*list(bb.children())[:-1], nn.Flatten())   # 输出 512
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 7), nn.Tanh(),
        )
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, state):
        # x: (B, 3*k*n_cam, H, W) → 每 3 通道一组（一帧一相机）做 ImageNet 归一化
        B, C, H, W = x.shape
        groups = self.k * self.n_cam
        x = x.view(B, groups, 3, H, W)
        x = (x - self.im_mean) / self.im_std
        x = x.view(B, 3 * groups, H, W)
        return self.head(torch.cat([self.vision(x), self.state(state)], dim=1))


def train(epochs=40, batch=64):
    from torch.utils.data import TensorDataset, DataLoader, random_split
    torch.manual_seed(7); np.random.seed(7)
    images, inputs, actions = load_fs_dataset(DATA_FILE)
    ds = TensorDataset(images, inputs, actions)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(7))
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    vloader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFSBC().to(device)
    # 预训练骨干小 lr、新 head 大 lr
    bb_params = [p for n, p in model.named_parameters() if "vision" in n]
    head_params = [p for n, p in model.named_parameters() if "vision" not in n]
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"[ResNet+FrameStack K={K}] 设备={device} 参数量={sum(p.numel() for p in model.parameters()):,} "
          f"训练={len(train_ds)} 验证={len(val_ds)} epochs={epochs}")
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0
        for x, s, y in loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(x, s), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(x)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            model.eval(); val = 0.0
            with torch.no_grad():
                for x, s, y in vloader:
                    val += loss_fn(model(x.to(device), s.to(device)), y.to(device)).item() * len(x)
            print(f"  epoch {ep:02d}/{epochs}  train_mse={total/len(train_ds):.6f}  val_mse={val/len(val_ds):.6f}", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "cam": CAM, "task": TASK_ID, "version": "resnet_fs", "K": K}, RESNET_FS_MODEL)
    print(f"已保存: {RESNET_FS_MODEL}")


def evaluate(episodes=10, save_video=True, tag="resfs"):
    import cv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFSBC().to(device)
    ckpt = torch.load(RESNET_FS_MODEL, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    env, suite, task = make_env(EVAL_CAM)
    inits = suite.get_task_init_states(TASK_ID)
    print(f"评测[{tag}] ResNet+FrameStack K={K} 任务: {task.language}")

    def run_group(group, idx0, n):
        succ, files = 0, []
        for i in range(n):
            tmp = os.path.join(HERE, "output", "_tmp_ep.mp4")
            writer = None
            if save_video:
                os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (EVAL_CAM, EVAL_CAM))
            env.reset()
            obs = env.set_init_state(inits[idx0 + i])
            buf = deque(maxlen=K)
            first = image(obs)
            if first.shape[0] != CAM:
                first = cv2.resize(first, (CAM, CAM), interpolation=cv2.INTER_AREA)
            for _ in range(K):
                buf.append(first)
            done = False; det = 0; step = 0
            for step in range(MAX_STEPS):
                img_full = image(obs)
                img = cv2.resize(img_full, (CAM, CAM), interpolation=cv2.INTER_AREA) if img_full.shape[0] != CAM else img_full
                buf.append(img)
                stacked = np.concatenate(list(buf), axis=2)
                s = np.concatenate((make_state(obs), find_bowl(img))).astype(np.float32)
                det += int(s[11] > 0.5)
                x = torch.from_numpy(stacked.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
                st = torch.from_numpy(s).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    a = model(x, st)[0].cpu().numpy()
                obs, reward, done, info = env.step(np.asarray(a, dtype=np.float32))
                if writer is not None:
                    writer.write(image(obs)[:, :, ::-1])
                if done:
                    break
            rate = det / max(1, step + 1)
            print(f"  [{tag}-{group}] init={idx0+i:02d}  steps={step+1:3d}  碗检出率={rate*100:.0f}%  success={done}", flush=True)
            if writer is not None:
                writer.release()
                if done:
                    dst = os.path.join(HERE, "output", f"success_{tag}_{group}_init{idx0+i:02d}.mp4")
                    os.replace(tmp, dst); files.append(dst)
                else:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            succ += int(done)
        return succ, files

    seen, _ = run_group("seen", 0, episodes)
    unseen, files = run_group("unseen", TRAIN_INITS, episodes)
    env.close()
    print(f"[{tag}] seen   成功率: {seen}/{episodes} = {seen/episodes*100:.0f}%")
    print(f"[{tag}] unseen 成功率: {unseen}/{episodes} = {unseen/episodes*100:.0f}%")
    for p in files:
        print("  成功录像: " + p)
    return seen, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    if args.mode == "train":
        train(args.epochs)
    elif args.mode == "eval":
        evaluate(args.episodes)


if __name__ == "__main__":
    main()
