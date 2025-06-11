# SD3RVD 分层强化学习框架

本项目实现了车辆与无人机协同的同日配送（Same-day Delivery with Drones and Vehicles, **SD3RVD**）问题的分层深度强化学习（Hierarchical DRL）框架。仓库中包含了环境模拟、上层/下层智能体、训练与评估流程等核心组件，便于复现和拓展相关研究。

## 目录结构

| 文件/目录 | 说明 |
|-----------|------|
| `env.py` | 定义环境 `SD3RVDEnv`，模拟客户请求生成、车队状态与决策周期等 |
| `agent_upper.py` | 实现上层 DQN 智能体，决定车辆在仓库等待还是出发 |
| `agent_lower.py` | 实现下层 PPO 智能体，对每个请求选择“无人机配送 / 车辆配送 / 拒绝” |
| `dataset_loader.py` | 生成训练/测试数据，支持不同的时间与空间分布 |
| `heuristics.py` | 提供插入式车辆路径规划和无人机 FIFO 分配等启发式算法 |
| `trainer.py` | 组织训练流程：环境交互、经验采集及两级代理更新 |
| `evaluation.py` | 在测试集上按确定性策略评估模型并统计指标 |
| `model.py` | 通用 MLP 构建与 DQN/Actor/Critic 网络结构 |
| `utils.py` | 日志、配置解析、随机种子设置、距离计算等工具函数 |
| `config.yaml` | 超参数与环境设置的统一配置文件 |
| `main.py` | 实验入口：加载配置并依次执行训练与评估 |

## 快速开始

1. **安装依赖**
   - 推荐 Python 3.8+。
   - 依赖主要包括 `PyTorch`、`gym`、`PyYAML` 等，可根据需要自行安装。

2. **运行训练**
   ```bash
   python main.py
   ```
   该命令会读取 `config.yaml`，初始化环境与两级智能体，随后开始训练并在指定周期进行评估。

3. **查看评估结果**
   训练结束后，脚本会自动调用 `Evaluation` 模块，输出平均服务量 (SN) 与服务率 (SR) 等指标。

## 算法框架与流程

本框架采用两层强化学习结构：

1. **上层调度 (UpperAgent)**：使用 DQN 根据环境概括状态判断车辆是等待还是立即出发。
2. **下层分配 (LowerAgent)**：使用 PPO 对每个待处理请求选择“无人机配送 / 车辆配送 / 拒绝”。
3. **环境交互**：`SD3RVDEnv.step` 接收两级决策，利用启发式方法更新车队和无人机行程，并返回奖励。
4. **训练循环**：`Trainer.train` 在每个 epoch 中循环执行上述决策过程，收集经验并触发两个智能体的更新，期间可按配置调用 `Evaluation` 进行评估。

流程概括如下：环境生成请求 → 下层代理为每个请求分配方案 → 上层代理决定是否派车 → 环境据此更新状态并给出奖励 → `Trainer` 保存经验并更新模型。

## 主要函数与数据结构

### `SD3RVDEnv`
- `reset()`：初始化时间、车队与请求事件，返回初始观测。
- `step(action)`：根据 `dispatch_decision` 和 `assignment_decisions` 更新仿真，输出 `(observation, reward, done, info)`。
- `vehicles` / `drones`：列表，每个元素包含 `id`、`status`、`route`、`available_time` 等字段。
- `request_buffer`：待处理请求列表，每个请求含 `id`、`arrival_time`、`deadline`、`location`。

### `UpperAgent`
- `select_action(state)`：ε-贪心选择等待或派车动作。
- `store_transition` 与 `update()`：维护经验回放并训练 DQN 网络。

### `LowerAgent`
- `select_action(request_state)`：对单个请求计算动作概率并采样，同时记录 log prob 与 value。
- `store_reward` 与 `update()`：累积奖励、计算优势并按 PPO 公式更新策略与价值网络。

### `Trainer`
- `train(num_epochs)`：核心训练循环协调环境和两级代理。
- `_extract_upper_state()` / `_extract_lower_state()`：从观测构建上层和下层所需的状态向量。

## 配置文件说明

`config.yaml` 包含实验的所有可调参数，例如训练轮数、学习率、车队规模、请求到达分布等。根据需要修改相应字段即可影响实验设置。常用字段示例：

```yaml
training:
  epochs: 400000
  evaluation_frequency: 1000
environment:
  working_hours: "10h"         # 每日工作时长
  vehicle_speed: 30            # 车辆速度 (km/h)
  drone_speed: 40              # 无人机速度 (km/h)
fleet:
  vehicles: 2
  drones: 3
```

## 代码阅读建议

1. **环境**：首先阅读 `env.py` 的 `reset` 和 `step` 函数，了解请求生成、车队状态更新与奖励计算方式。
2. **代理**：查看 `agent_upper.py` 和 `agent_lower.py`，理解 DQN 与 PPO 的实现细节以及状态/动作定义。
3. **训练流程**：在 `trainer.py` 中跟踪 `train` 方法，熟悉一轮交互的数据流和模型更新顺序。
4. **启发式方法**：`heuristics.py` 中的 `cheapest_insertion` 与 `FIFO_assignment` 函数演示了车辆与无人机的调度逻辑，可与学习策略结合使用。

## 扩展与实验

- 调整 `config.yaml` 中的超参数或数据分布，观察训练与评估结果的变化。
- 替换或修改网络结构（位于 `model.py`）以尝试新的算法变体。
- 环境设计中如需新增约束或状态特征，可在 `env.py` 和相应的状态提取函数中修改。

## 许可证

本仓库未包含明确许可证。如需在科研或商业场景中使用，请在遵循原作者意愿的前提下继续完善相关声明。

