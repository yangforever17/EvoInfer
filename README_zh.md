# EvoInfer

**推理系统的智能体自进化。** EvoInfer 让编程智能体（Codex、Claude Code）持续地自行改进一个大模型推理引擎：

- 智能体提出修改；
- 一个可信的适应度函数决定哪些修改留下；
- 被接受的版本成为下一代的起点；
- 智能体所依据的知识也随之进化。

[English](README.md)

我们在 Qwen3.5-35B-A3B 上做了实验。这个模型混合了门控 DeltaNet 与门控注意力，用 256 专家的 MoE，并带 MTP 投机解码。在单块 A100 上，约 20 小时的智能体时间把这个模型在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 里的朴素移植版，从 **17.0 提高到 297.5 解码 tok/s**，端到端吞吐达到 **默认配置 vLLM 的 1.73–1.85 倍**。每个版本都逐 token 地与冻结的参考实现比对过。

| | 结果 |
|---|---|
| 主线（20 智能体小时，Codex） | 解码 17.0 → 297.5 tok/s（17.5×）；端到端为默认 vLLM 的 1.73× / 1.85×；15 个版本中 7 个来自重组 |
| 长上下文谱系（8K–31K 提示） | 通用引擎相对默认 vLLM 的优势降到 1.02×；进化两代（改动 34 行）后恢复到 1.60× |
| 端侧谱系（32 GB 显卡，4 比特专家） | 在同一份权重上为默认 vLLM 的 1.61× / 1.79× |
| 同一 harness 换策略模型（3 轮 × 2 智能体） | Qwen3.8-Max 6.43×，Codex 6.32×（3 次均值），GLM-5.3 4.31×，DeepSeek-V4-Pro 1.52× |
| 换目标模型、换种子引擎 | OLMoE-1B-7B 2.24×，Gemma 4 26B-A4B 9.12×，mini-sglang 种子 1.82× |

进化出的引擎在 [`engines/`](engines/) 里，可以直接用 `orchestrator.py eval` 测。

## 工作方式

- **个体层**：一个 episode 是一个智能体在预算内改进当前最优引擎。预算例如 75 分钟、8 次评测；方向提示有 `overhead`、`moe`、`spec`、`free` 等。智能体只能通过 `evo` 命令接触 GPU，包括 `evo eval`、`evo run -- <命令>`、`evo history`、`evo board`、`evo claim`、`evo submit`。
- **适应度**：
  - 正确性在候选自己的生成轨迹上判定：对每个生成 token，在相同前缀下计算参考实现的最优 logit 与所选 token 的 logit 之差。浮点重排造成的近似平局可以通过，真实错误不能。
  - 性能用配对测量：在同一块 GPU 上紧接候选之前重测父版本。
  - 接受要求开发集和智能体从未见过的验收集同时有增益。
- **种群层**：
  - episode 并行运行；
  - 每轮结束时合并被接受的兄弟改动，冲突时交给合并智能体；
  - 低于噪声阈值的正确改动入池，捆绑后整体评测；
  - 一块共享黑板防止智能体改同一批函数。
- **元层**：元智能体依据记录下来的证据改写智能体的技能（`skills/`），但不能碰适应度函数。
- **场景**：每个谱系为一个部署场景进化，例如长上下文、端侧、另一个模型。

细节见 [docs/architecture.md](docs/architecture.md)。

## 快速上手

需要：

- **GPU 机器**：Linux，一块 80 GB 显卡（A100 / H100，用于 BF16 的 Qwen3.5-35B-A3B），CUDA 13，Python 3.12。
- **智能体一侧**：Python ≥ 3.11、git、rsync，以及已登录的 [Codex CLI](https://github.com/openai/codex) 或 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)。

两侧可以是同一台机器（默认），也可以通过 SSH 连接两台机器。

```bash
git clone <本仓库> evoinfer && cd evoinfer
bash scripts/setup_gpu_host.sh                # 评测用的 Python 环境，位于 ~/.evoinfer/venv
cp evoinfer.example.toml evoinfer.toml        # 填 [models] qwen3_5 和 [gpus] pool
python3 orchestrator.py doctor                # 检查全部依赖，并告诉你缺什么、怎么补
python3 orchestrator.py setup                 # 引擎存档、判官引擎、提示集
```

先测一下论文里的结果：

```bash
python3 orchestrator.py eval engines/evo-015 --gpu 0      # 完整评测 + 判官，约 10 分钟
```

自己从朴素种子开始进化：

```bash
python3 orchestrator.py init --scenario base --from baseline --gpu 0   # 评测种子，作为主线的根
python3 orchestrator.py evolve --rounds 3 --gpus 0,1                    # 每轮每块 GPU 一个智能体，然后整合
python3 orchestrator.py status                                          # 当前最优版本和全部实验记录
```

`evolve` 默认用 Codex。`--agent claude` 改用 Claude Code；`--policy <名字>` 使用配置文件里定义的策略。

- **用时**：一轮（两个 75 分钟的 episode 加整合）约 1.5–2 小时。
- **token**：一个 episode 读 3–5 M 输入 token，大部分是缓存。
- **记录**：智能体的完整对话在 `runs/<episode>/`；所有评测、判定和存档在 GPU 机器上的 EvoInfer 目录里（`~/.evoinfer/records`）。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `orchestrator.py` | episode、终评与判定、整合（合并、捆绑）、谱系，以及 `doctor` / `setup` / `evolve` / `eval` |
| `meta.py` | 进化技能的元智能体 |
| `bin/evo` | 智能体调用的命令；通过工作区里的队列把每次调用交给编排器 |
| `framework/evo/` | 在 GPU 机器上运行：`evo` 命令、评测器、基准、判官、场景、协调黑板 |
| `skills/v0/` | 智能体读的手写技能（论文的初始知识） |
| `engines/` | 种子引擎和论文中进化出的引擎 |
| `data/` | 提示集：`workloads/`（开发集，智能体可见）和 `holdout/`（只用于验收） |
| `experiments/` | 论文实验的脚本，含无框架基线 |
| `tests/` | 不需要 GPU 的测试（`python3 -m pytest tests`） |

更多文档：

- [配置](docs/configuration.md)
- [扩展：新场景、新模型、新种子引擎](docs/extending.md)
- [复现论文](experiments/README.md)

## 安全说明

智能体会在你的 GPU 上运行自己写的代码。EvoInfer 限制了它们能接触的范围：

- Codex 在 `--sandbox workspace-write` 下运行，Claude Code 只能用白名单里的工具。
- 评测器、判官和验收集都在智能体的工作区之外。
- 智能体拿不到 GPU 机器的 SSH 权限或 API 密钥。

即便如此，请在允许这样做的机器和账号上运行，并检查对话记录。

## 许可

MIT，见 [LICENSE](LICENSE)。`engines/` 中的引擎派生自 nano-vllm（MIT），保留了它的许可文件。
