# SmolVLA WaFT

在 SmolVLA 的 WaFM 动作头上，对 LIBERO H10 动作块的二层 Haar 低频分量 A2 提前去噪。实现通过 LeRobot 的 `SmolVLAPolicy.forward` 进入动作域和小波域训练损失。已有 checkpoint 未设置 `wavelet_a2_schedule_alpha` 时，默认使用原 DWT 的线性路径。

## A2 去噪路径

流时间从 $t=1$ 的高斯噪声走向 $t=0$ 的示范动作。设 $a$ 为示范动作、$\epsilon$ 为高斯噪声、$P$ 为选定 A2 分量的正交投影。启用后使用下式构造训练输入和速度目标。

$$
\begin{aligned}
s(t) &= t-\alpha\sin^2(\pi t)/\pi, \\
x_t &= a+t(\epsilon-a)+(s(t)-t)P(\epsilon-a), \\
u_t &= \epsilon-a-\alpha\sin(2\pi t)P(\epsilon-a).
\end{aligned}
$$

- $\alpha=0.30$ 时，$t=0.5$ 的 A2 噪声权重约为 0.405。两个端点仍是原高斯噪声和示范动作。
- $P$ 只覆盖第 1–4 帧、第 5–8 帧的机械臂前六个动作维度。每个四帧块用区块均值提取 A2 分量，块内出现 episode padding 时，该块沿用线性路径。
- D1、D2、夹爪和第 9、10 帧沿用线性路径。原 DWT 复制第 10 帧两次计算第三个 A2，重建后裁回 10 帧。
- 推理对动作速度场做 10 步 Euler 积分，生成 10 帧且执行 10 帧后重新规划（H10/E10）。

## 训练

此仓库以源码覆盖的方式接入 [LeRobot](https://github.com/huggingface/lerobot)。使用训练环境对应的 LeRobot 源码检出和 SmolVLA 依赖。该检出需要提供 `lerobot.policies.common.flow_matching`、`lerobot.policies.common.vla_utils` 和 SmolVLA 的 RTC 接口。

在独立的 LeRobot 源码检出中安装 SmolVLA 依赖，然后从本仓库目录复制策略文件。下列命令使用 Bash。

```bash
export LEROBOT_DIR=/path/to/lerobot
cp src/lerobot/policies/smolvla/*.py "$LEROBOT_DIR/src/lerobot/policies/smolvla/"
python -m pip install -e "$LEROBOT_DIR[smolvla]"
```

在 [LIBERO 训练配置](configs/libero_h10_a2.json) 的副本中，将 `policy.pretrained_path` 填为你的 SmolVLA LIBERO 初始 checkpoint。可按环境设置 `dataset.root` 和 `policy.vlm_model_name`，后者默认指向 Hugging Face 的 SmolVLM2。配置使用 Haar L2、FiLM 128、预训练线性读出初始化、小波损失权重 0.1、batch size 64 和 seed 1000。

```bash
lerobot-train --config_path=path/to/libero_h10_a2_local.json
```

`policy.wavelet_a2_schedule_alpha=0.3` 从第一步启用提前去噪。设为 `0.0` 可运行原 DWT 对照。配置保留 25k 学习率计划，每 5k 保存 checkpoint，20k 对照取 `checkpoints/020000/pretrained_model`。推理加载相应 checkpoint 后沿用 LeRobot 的 SmolVLA 入口。

## 已完成的实验

先从同一原 DWT 15k checkpoint 配对续训 5k，并在累计 20k 时比较。单训练种子下，原 DWT 成功 141/200，A2 提前去噪成功 144/200（McNemar 配对检验 $p=0.743$）。采样中段的 A2 状态误差下降 14.8%，闭环收益仍不确定。

从第 1 步启用 A2 调度的实验使用原训练保存的 DWT 20k checkpoint 作直接对照。上面的闭环数字来自 15k→20k 续训。

SmolVLA 基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，小波头基于 [WaFM](https://github.com/Ruyi231/Wafm) 的 PyTorch 移植。许可见 [LICENSE](LICENSE)。
