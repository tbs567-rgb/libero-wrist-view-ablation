# -*- coding: utf-8 -*-
"""PPT 补强用图表 v2：fig7 案例帧对比 / fig8 管线架构 / fig9 全成绩表。
输出：figs/fig7_case_obj_t9.png, figs/fig8_pipeline.png, figs/fig9_results_table.png
用法：./.venv/Scripts/python.exe make_figs_v2.py
"""
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_POS = "#2b6cb0"
C_NEG = "#c53030"
C_MID = "#8B97A8"
C_SP = "#2b6cb0"
C_OB = "#dd6b20"
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BD = "C:/Windows/Fonts/msyhbd.ttc"


def get_font(sz, bold=False):
    return ImageFont.truetype(FONT_BD if bold else FONT, sz)


# ---------------------------------------------------------------- fig7
def fig7_case():
    """案例帧对比：单相机（从不接近）vs 双相机（成功）。来源 = 已录制的冒烟案例视频。"""
    src_s = "videos_cases/obj_t9_ep00_never-approach_single.mp4"
    src_d = "videos_cases/obj_t9_ep00_never-approach_dual.mp4"

    def frames(path, n=6):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, n).astype(int)
        out = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            out.append((int(i), Image.fromarray(f)))
        cap.release()
        return out, total

    fs, ts = frames(src_s)
    fd, td = frames(src_d)
    print(f"single: {ts} 帧; dual: {td} 帧")

    FRAME, GAP, M = 180, 6, 14
    W = M * 2 + 6 * FRAME + 5 * GAP
    strip_h, cap_h = 30, 30
    H = 12 + strip_h + FRAME + 12 + strip_h + FRAME + cap_h
    canvas = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(canvas)

    def draw_strip(y, text, bg, fg):
        d.rounded_rectangle([M, y, W - M, y + strip_h - 6], radius=6, fill=bg)
        d.text((M + 12, y + 5), text, font=get_font(15, True), fill=fg)

    def draw_row(y0, items):
        x = M
        for i, (idx, im) in enumerate(items):
            im = im.resize((FRAME, FRAME), Image.LANCZOS)
            canvas.paste(im, (x, y0))
            d.rectangle([x, y0, x + FRAME - 1, y0 + FRAME - 1], outline="#B8C2CE", width=1)
            tag = f"步 {idx}"
            tw = d.textlength(tag, font=get_font(13))
            d.rectangle([x + FRAME - tw - 14, y0 + FRAME - 22, x + FRAME - 2, y0 + FRAME - 2],
                        fill=(255, 255, 255))
            d.text((x + FRAME - tw - 8, y0 + FRAME - 21), tag, font=get_font(13), fill="#333333")
            x += FRAME + GAP

    y = 6
    draw_strip(y, "单相机：悬停原地 · 300 步超时 —— 从未进入抓取邻域（失败）", "#FDECEA", C_NEG)
    y += strip_h
    draw_row(y, fs)
    y += FRAME + 12
    draw_strip(y, "双相机：接近 → 抓取 → 运送入篮 · 122 步成功（同一起始态）", "#E8EFF8", "#1E4FA8")
    y += strip_h
    draw_row(y, fd)
    y += FRAME
    d.text((M + 2, y + 5),
           "案例：LIBERO-Object t9 橙汁 · 初始态 #0（逐帧抽取；完整录像见附录视频集）",
           font=get_font(13), fill="#8B97A8")
    outp = os.path.join(OUT, "fig7_case_obj_t9.png")
    canvas.save(outp)
    print("[OK]", outp, canvas.size)


# ---------------------------------------------------------------- fig8
def fig8_pipeline():
    """训练与推理管线图（含真实画面；imshow 用 aspect='auto' 防止坐标系被改写）。"""
    def grab(path, idx):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        cap.release()
        assert ok, path
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    av = grab("videos_cases/obj_t9_ep00_never-approach_dual.mp4", 50)
    wr = grab("videos_cases/obj_t9_ep00_never-approach_dual_wrist.mp4", 50)

    fig = plt.figure(figsize=(12.8, 4.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_autoscale_on(False)
    ax.axis("off")

    def box(x, y, w, h, fc="#F7F9FC", ec="#B9C7DA", r=2.0, lw=1.4, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def txt(x, y, s, sz=11, c="#1A2230", w="normal", ha="center", va="center", z=5):
        ax.text(x, y, s, fontsize=sz, color=c, fontweight=w, ha=ha, va=va, zorder=z)

    def arrow(x1, y1, x2, y2, c="#5B7EA8", lw=1.8, style="-|>", rad=0.0, z=4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                     color=c, lw=lw, zorder=z,
                                     connectionstyle=f"arc3,rad={rad}"))

    # ---- 输入（真实画面，宽 11.5 高 36 ≈ 正方形）----
    ax.imshow(av, extent=[3, 14.5, 52, 88], aspect="auto", zorder=1)
    ax.add_patch(plt.Rectangle((3, 52), 11.5, 36, fc="none", ec="#5B7EA8", lw=1.6, zorder=2))
    ax.imshow(wr, extent=[3, 14.5, 9, 45], aspect="auto", zorder=1)
    ax.add_patch(plt.Rectangle((3, 9), 11.5, 36, fc="none", ec="#C68A3F", lw=1.6, zorder=2))
    txt(8.75, 90.8, "第三人称 agentview", 10.5, "#4A5568")
    txt(8.75, 47.6, "腕部 eye_in_hand", 10.5, "#8C5A1E")

    # ---- 顶部注记 ----
    txt(2.6, 97.5, "输入：双路 96×96 RGB · 4 帧堆叠；对照组仅 agentview 通道（12ch）",
        9.5, "#8B97A8", ha="left")
    txt(98.5, 97.5, "总参数 ≈ 11.9M · 单卡 6GB 可训", 9.5, "#8B97A8", ha="right")

    # ---- 主干三框 ----
    box(27, 44, 20, 48)
    txt(37, 85.5, "ResNet-18 骨干", 13, "#0E3F8C", "bold")
    box(28.3, 48.5, 17.4, 30, fc="#FFFFFF", ec="#D6DCE5", r=1.2, lw=1.0)
    txt(37, 72.5, "ImageNet 预训练权重", 9.5, "#4A5568")
    txt(37, 66.5, "conv1 扩至 24 通道", 9.5, "#4A5568")
    txt(37, 60.5, "（4 帧 × 2 相机）", 9.5, "#4A5568")
    txt(37, 46.2, "→ 512 维视觉特征", 10.5, "#1E4FA8", "bold")

    box(50, 44, 18, 48)
    txt(59, 85.5, "特征拼接", 13, "#0E3F8C", "bold")
    box(51.3, 48.5, 15.4, 30, fc="#FFFFFF", ec="#D6DCE5", r=1.2, lw=1.0)
    txt(59, 71.5, "视觉 512", 9.5, "#4A5568")
    txt(59, 65.5, "＋ 本体状态 64", 9.5, "#4A5568")
    txt(59, 59.5, "＝ 576 维", 10, "#1E4FA8", "bold")
    txt(59, 46.2, "→ MLP 决策头", 10.5, "#1E4FA8", "bold")

    box(73, 44, 17, 48)
    txt(81.5, 85.5, "动作输出", 13, "#0E3F8C", "bold")
    box(74.3, 48.5, 14.4, 30, fc="#FFFFFF", ec="#D6DCE5", r=1.2, lw=1.0)
    txt(81.5, 72.5, "256 → 128 → 7", 9.5, "#4A5568")
    txt(81.5, 66.5, "tanh 有界输出", 9.5, "#4A5568")
    txt(81.5, 60.5, "Δ位姿 6 + 夹爪 1", 9.5, "#4A5568")
    txt(81.5, 46.2, r"$a_t\in\mathbb{R}^7$", 12, "#1E4FA8", "bold")

    # ---- 状态盒 ----
    box(51, 10, 20, 22, fc="#FFF6EE", ec="#E8C9A8")
    txt(61, 26.5, "8 维本体状态", 11, "#8C5A1E", "bold")
    txt(61, 20.0, "末端位置/姿态/夹爪", 9, "#8C5A1E")
    txt(61, 14.2, "（与评测协议同源）→ 64 维", 9, "#8C5A1E")

    # ---- 箭头 ----
    arrow(15.3, 70, 26.5, 70)                 # agentview → 骨干
    arrow(15.3, 27, 26.5, 55, rad=-0.18)      # 腕部 → 骨干（斜）
    arrow(47, 70, 49.5, 70)                   # 骨干 → 拼接
    arrow(68, 70, 72.5, 70)                   # 拼接 → 动作
    arrow(61, 32.5, 61, 43.5)                 # 状态 → 拼接
    # 闭环
    ax.plot([81.5, 81.5], [43.5, 3.0], color="#5B7EA8", lw=1.8, zorder=4)
    ax.plot([81.5, 8.75], [3.0, 3.0], color="#5B7EA8", lw=1.8, zorder=4)
    arrow(8.75, 3.0, 8.75, 8.4)
    txt(54, 5.6, "闭环：每步执行 1 次动作 → 获取新观测（无规划器 / 无未来预测）",
        10.5, "#4A5568")

    fig.savefig(os.path.join(OUT, "fig8_pipeline.png"), dpi=200)
    plt.close(fig)
    print("[OK]", os.path.join(OUT, "fig8_pipeline.png"))


# ---------------------------------------------------------------- fig9
def fig9_table():
    """14 任务全成绩表（含两个套件平均）。数据源 = 权威 JSON。"""
    sp_s = json.load(open("benchmark_results.json", encoding="utf-8"))
    sp_d = json.load(open("benchmark_results_dual.json", encoding="utf-8"))
    ob = json.load(open("object_check_results.json", encoding="utf-8"))
    SP_NAMES = ["0 盘盅之间", "1 烤盅旁", "2 桌中央", "3 盒上", "4 抽屉",
                "5 烤盅顶", "6 盒旁", "7 炉灶", "8 盘旁", "9 柜顶"]
    OB_NAMES = {0: "汤罐", 3: "酱瓶", 6: "黄油", 9: "橙汁"}

    rows = []
    for t in range(10):
        rows.append(["sp", f"sp{t} {SP_NAMES[t]}", sp_s[f"task{t}"]["rate"],
                     sp_d[f"task{t}"]["rate"]])
    sp_avg_s = np.mean([r[2] for r in rows])
    sp_avg_d = np.mean([r[3] for r in rows])
    rows.append(["spavg", "Spatial 平均（10 任务）", sp_avg_s, sp_avg_d])
    ob_rows = []
    for t in (0, 3, 6, 9):
        ob_rows.append(["ob", f"ob{t} {OB_NAMES[t]}", ob[f"task{t}|single"]["rate"],
                        ob[f"task{t}|dual"]["rate"]])
    ob_s = [r[2] for r in ob_rows]; ob_d = [r[3] for r in ob_rows]
    rows += ob_rows
    rows.append(["obavg", "Object 平均（抽检 4 任务）", np.mean(ob_s), np.mean(ob_d)])

    n = len(rows)
    fig = plt.figure(figsize=(13.2, 5.35))
    ax = fig.add_axes([0.012, 0.035, 0.976, 0.955])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    top, bot = 100.0, 2.0
    hdr_h = 100 * 0.055
    rowh = (top - bot - hdr_h) / n
    z_ = 1

    def cell_bg(y0, y1, color):
        ax.add_patch(plt.Rectangle((0, y0), 100, y1 - y0, fc=color, ec="none", zorder=z_))

    y0 = top - hdr_h
    cell_bg(y0, top, "#0E3F8C")
    ax.text(2.0, y0 + hdr_h / 2, "任务（LIBERO 双套件 · 14 项）", fontsize=12.5, color="#FFFFFF",
            fontweight="bold", va="center", ha="left")
    ax.text(62.0, y0 + hdr_h / 2, "单相机 %", fontsize=12.5, color="#BFD3EE", va="center", ha="right")
    ax.text(80.6, y0 + hdr_h / 2, "双相机 %", fontsize=12.5, color="#BFD3EE", va="center", ha="right")
    ax.text(98.8, y0 + hdr_h / 2, "Δ pp", fontsize=12.5, color="#FFC107", fontweight="bold",
            va="center", ha="right")

    y = y0
    for i, (tag, name, s, dd) in enumerate(rows):
        g = dd - s
        y1 = y - rowh
        if tag == "spavg" or tag == "obavg":
            bg = "#E8EFF8"
        elif tag == "ob":
            bg = "#FDF4EA" if i % 2 == 0 else "#FBF0E3"
        else:
            bg = "#FFFFFF" if i % 2 == 0 else "#F7F9FC"
        cell_bg(y1, y, bg)
        dot_c = C_SP if tag.startswith("sp") else C_OB
        ax.plot(2.6, (y + y1) / 2, marker="o", ms=4.6, color=dot_c, zorder=4)
        bold = "bold" if tag.endswith("avg") else "normal"
        ax.text(4.6, (y + y1) / 2, name, fontsize=11.2, color="#1A2230", fontweight=bold,
                va="center", ha="left")
        ax.text(62.0, (y + y1) / 2, f"{s:.1f}", fontsize=11.2, color="#333333",
                fontweight="bold", va="center", ha="right")
        ax.text(80.6, (y + y1) / 2, f"{dd:.1f}", fontsize=11.2, color="#0E3F8C",
                fontweight="bold", va="center", ha="right")
        if g > 4.5:
            gc = C_POS
        elif g < -4.5:
            gc = C_NEG
        else:
            gc = C_MID
        sign = "+" if g >= 0 else ""
        ax.text(98.8, (y + y1) / 2, f"{sign}{g:.1f}", fontsize=11.2, color=gc,
                fontweight="bold", va="center", ha="right")
        y = y1
        ax.plot([1.2, 98.8], [y, y], color="#E3E8EF", lw=0.6, zorder=3)

    fig.text(0.5, 0.006, "灰 = |Δ| ≤ 4pp（运行波动带，≈2 局）；数据：256 渲染评测批次，每任务 50 局",
             fontsize=10, color="#8B97A8", ha="center", va="bottom")
    outp = os.path.join(OUT, "fig9_results_table.png")
    fig.savefig(outp, dpi=200)
    plt.close(fig)

    pos = sum(1 for r in rows for g in [r[3] - r[2]] if not r[0].endswith("avg") and g > 4.5)
    neg = sum(1 for r in rows for g in [r[3] - r[2]] if not r[0].endswith("avg") and g < -4.5)
    mid = sum(1 for r in rows for g in [r[3] - r[2]] if not r[0].endswith("avg") and -4.5 <= g <= 4.5)
    print(f"[复核] Spatial 均: {sp_avg_s:.1f}→{sp_avg_d:.1f} (+{sp_avg_d - sp_avg_s:.1f})")
    print(f"[复核] Object 均:  {np.mean(ob_s):.1f}→{np.mean(ob_d):.1f} (+{np.mean(ob_d) - np.mean(ob_s):.1f})")
    print(f"[复核] 符号: {pos} 正 / {mid} 波动带 / {neg} 负")
    print("[OK]", outp)


if __name__ == "__main__":
    fig7_case()
    fig8_pipeline()
    fig9_table()
    import shutil
    dst = "腕部视角增益条件与机制_汇报/assets"
    for f in ("fig7_case_obj_t9.png", "fig8_pipeline.png", "fig9_results_table.png"):
        shutil.copy(os.path.join(OUT, f), os.path.join(dst, f))
    print("[OK] 已同步到 PPT assets/")
