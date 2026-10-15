# 腕部视角的增益条件与机制：LIBERO 跨套件双视角消融

**Wrist-View Gain in Behavior Cloning: Conditions and Mechanisms — A Cross-Suite Dual-View Ablation on LIBERO**

> 用一张 6GB 显卡回答一个全行业默认、却从未被检验的问题：机械臂的第二只眼睛（腕部相机），到底值几分？什么时候值？为什么值？

---

## 这是什么

一个**受控消融实验研究**：在 LIBERO 标准基准上，训练三个"只差相机"的孪生模型（仅全局 / 仅腕部 / 双相机），用官方协议评测 1400+ 局，并逐局分析成败原因。

- 模型：ResNet-18 + 4 帧堆叠 + 本体状态，**1138 万参数**，单任务训练 **35 分钟**（6GB 显卡）
- 评测：每任务 50 个官方初始态 × 300 步上限 × 256 渲染，仿真系统自动判卷
- 分析：逐局四阶段（接近/抓取/运输/放置）分解 + 失败自动标签 + 冻结/重放干预实验 + 多种子复现

## 核心发现

| 发现 | 内容 |
|---|---|
| **可预测的增益规律** | r(增益, 单相机基线) = **−0.944**（14 任务）：单相机越差，加腕部相机赚得越多 |
| **机制 = 抓取对准** | 抓取失败 129→34 局（**−74%**），放置环节 37→38 **零改善**；瞄准误差 6.6→5.5cm（r=−0.72） |
| **负增益再解释** | 单种子口径的负增益（t7 −28pp）经多种子检验由基线种子方差主导（94/2/44%），均值口径下为 +35.3pp |
| **三配置三种模式** | 冗余型（t2：腕 98% ≈ 双 100%）/ 互补型（ob9：6/10/94，缺一不可）/ 干扰型（t7：拼接反而最差） |
| **测量方法论** | 单侧"冻结"式消融会系统性夸大依赖（10 格中 **8 格为伪影**）；提出"冻结+重放"双对照协议 |
| **测量规范** | 渲染档位是第一类混淆变量（64/128/256 = 40/78/76）；评测运行波动 ±2 局级 |

## 仓库结构

```
├── low_resource_official.py    # 主脚本：train/eval（--dual-cam / --wrist-cam / --seed 开关）
├── low_resource_resnet_fs.py   # 模型定义（ResNetFSBC）
├── low_resource_bc.py          # 环境/图像/常量共享模块
├── low_resource_framestack.py  # 帧堆叠 BC（前期自造数据线）
├── stage_analysis.py           # 机制探针：四阶段分解 + 失败标签 + 冻结/重放干预
├── init_pose_analysis.py       # 开局级几何分析（n=700）
├── run_object_check.py         # 第二套件（Object）批跑
├── run_wrist_check.py          # 仅腕部三组对照批跑
├── run_multiseed.py            # 多种子复现批跑
├── make_figs.py / make_figs_v2.py  # 论文图表生成
├── data_audit.py               # 从原始 json 现算复核全部关键数字
├── results/                    # 全部实验结果 json（可被 data_audit.py 复核）
├── figures/                    # fig1–fig9（论文/汇报用图）
├── docs/                       # 论文初稿、数据溯源手册、研究大纲、学位论文章节
└── assets/videos/              # 7 组失败案例单|双侧拼视频
```

## 快速开始（零基础）

```bat
start.bat        # Windows：双击即有菜单（自检/训练/评测/探针/图表/复核）
```

```bash
./start.sh       # Linux / macOS 等价入口
```

命令行方式：

```bash
# 1. 环境（Python 3.9+，Windows/Linux 均可；MuJoCo 需先于 torch 初始化）
pip install mujoco==3.3.7 robosuite==1.4.0 torch torchvision opencv-python h5py numpy matplotlib imageio imageio-ffmpeg
# LIBERO 与数据集（官方 HuggingFace，国内用 hf-mirror）：
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && pip install -e LIBERO
# 数据集放到 ./datasets/libero_spatial 与 ./datasets/libero_object

# 2. 训练一个双相机模型（约 35 分钟 / 6GB 显卡）
python low_resource_official.py train --h5 datasets/libero_spatial/<任务>_demo.hdf5 --epochs 600 --dual-cam

# 3. 官方协议评测（50 局，256 渲染）
LIBERO_EVAL_CAM=256 python low_resource_official.py eval --h5 <同上> --task-id 0 --dual-cam

# 4. 仅腕部 / 换种子：把 --dual-cam 换成 --wrist-cam，或加 --seed 1
```

每个论文数字的复现命令见 **`docs/数据溯源手册.md`**；改任何数字前先跑 `python data_audit.py` 复核。

## 实验规模

- **1400+ 局**主评测（14 任务 × 2 配置 × 50 开局）
- **700** 个开局几何分析、**500 局**机制探针、**200 局**跨套件画像
- **8 单元**多种子复现、**10 格**干预对照、**3 单元**仅腕部对照
- 全部断点续跑，全部数字可由脚本现算复核

## 诚实边界

- 不声称超越任何方法（1138 万参数 BC 与 7B 级 VLA 不同赛道）
- Object 为 4/10 抽检；仅仿真无真机；融合方式为刻意朴素的通道拼接
- 仅腕部数字为单种子；t7"干扰型"判读需多种子复核（列入未来工作）

## 致谢

基于 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（NeurIPS 2023）、robosuite、MuJoCo。相关工作：Look Closer（RA-L 2022）、VLA-Adapter、CronusVLA / SemanticVLA（AAAI 2026）、Visible Touch（2026）。

## License

MIT（代码部分）。文档与图表同样以 MIT 发布；LIBERO 数据集版权归原作者所有。
