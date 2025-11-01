# Huawei CodeCraft 2017 CDN Optimization with OpenEvolve

## 问题概述

这是华为 CodeCraft 2017 编程大赛的题目 - 一个视频内容分发网络(CDN)服务器部署优化问题。

**目标**: 最小化总成本 = 服务器部署成本 + 带宽租用成本

**约束条件**:
- 满足所有消费节点的视频带宽需求
- 不超过网络链路的带宽容量
- 不超过服务器的输出能力上限

**问题本质**: 这是一个设施选址(Facility Location)和多商品流(Multi-Commodity Flow)的组合优化问题，属于NP-hard问题。

## OpenEvolve 原理

OpenEvolve 使用**进化算法** + **大语言模型**来优化代码：

1. **MAP-Elites算法**: 维护程序多样性，在特征空间中保留精英解
2. **Island-Based Evolution**: 多个独立种群并行演化，定期迁移交流
3. **LLM驱动变异**: 使用Qwen等大模型生成智能代码变异
4. **级联评估**: 多阶段评估策略，快速过滤低质量解
5. **进程池并行**: 并行评估多个程序，加速演化

## 文件说明

- `initial_program.py` - 初始基线解决方案（简单贪心策略）
- `evaluator.py` - 评估器，测试程序在多个案例上的表现
- `config.qwen.yaml` - Qwen模型配置文件
- `case_example/` - 测试用例目录
  - `batch1/0Primary/` - 初级测试用例
  - `batch1/1Intermediate/` - 中级测试用例
  - `batch1/2Advanced/` - 高级测试用例
  - `batch2/`, `batch3/` - 更多测试批次

## 快速开始

### 1. 安装依赖

```bash
cd /path/to/openevolve
pip install -e ".[dev]"
```

### 2. 验证初始程序

```bash
cd examples/huawei_2017

# 测试单个案例
python initial_program.py case_example/batch1/0Primary/case0.txt

# 完整评估
python -c "from evaluator import evaluate; result = evaluate('initial_program.py'); print(result.metrics)"
```

预期输出:
```
{'valid_solutions': 1.0, 'avg_cost': 149920.0, 'combined_score': 0.6267, ...}
```

### 3. 运行进化优化

**Linux/Mac**:
```bash
chmod +x run_evolution.sh
./run_evolution.sh --iterations 300
```

**Windows**:
```batch
run_evolution.bat --iterations 300
```

**Python直接调用**:
```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
    --config config.qwen.yaml \
    --iterations 300
```

### 4. 查看结果

进化过程中会自动保存：

- `openevolve_output/best/best_program.py` - 最优解
- `openevolve_output/best/best_program_info.json` - 最优解的性能指标
- `openevolve_output/checkpoints/checkpoint_N/` - 定期保存的检查点
- `openevolve_output/logs/` - 详细日志

### 5. 从检查点恢复

如果进化中断，可以从检查点恢复：

```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
    --config config.qwen.yaml \
    --checkpoint openevolve_output/checkpoints/checkpoint_150 \
    --iterations 50
```

## 性能指标说明

评估器返回以下指标：

- `valid_solutions` (0.0-1.0): 有效解的比例
- `avg_cost`: 平均总成本（越低越好）
- `cost_score` (0.0-1.0): 成本得分（归一化）
- `time_score` (0.0-1.0): 执行时间得分
- `combined_score` (0.0-1.0): 综合得分
  - 公式: `0.5 * valid_rate + 0.4 * cost_score + 0.1 * time_score`

**基线性能** (初始程序):
- Valid Solutions: 100%
- Average Cost: ~150,000
- Combined Score: ~0.63

**优化目标**:
- 降低平均成本到 100,000 以下
- 提高 Combined Score 到 0.75+

## 配置参数说明

`config.qwen.yaml` 中的关键参数：

```yaml
max_iterations: 300          # 进化迭代次数
checkpoint_interval: 30      # 每30次迭代保存检查点

llm:
  primary_model: "Qwen/Qwen2.5-Coder-32B-Instruct"  # 使用的模型
  temperature: 0.8           # 生成的随机性(0-1，越高越随机)
  max_tokens: 8192           # 单次生成的最大token数

database:
  num_islands: 4             # 岛屿数量（并行演化的种群数）
  population_size: 80        # 种群大小
  elite_selection_ratio: 0.25  # 精英选择比例

evaluator:
  timeout: 90                # 单个程序评估超时(秒)
  parallel_evaluations: 3    # 并行评估的程序数量
```

**调优建议**:
- 如果想快速实验: `max_iterations: 100`, `population_size: 40`
- 如果想更好结果: `max_iterations: 500`, `population_size: 120`, `num_islands: 6`
- 如果API调用受限: 降低 `parallel_evaluations` 和 `num_islands`

## 算法优化方向

OpenEvolve 的 LLM 会尝试探索以下优化策略：

1. **贪心启发式**
   - 服务器放置策略（靠近高需求区域）
   - 服务器类型选择（容量vs成本权衡）

2. **局部搜索**
   - 服务器重定位
   - 服务器增删
   - 需求重分配

3. **流量优化**
   - 更好的路由算法
   - 带宽成本最小化
   - 负载均衡

4. **高级技术**
   - 聚类消费节点
   - 分层部署策略
   - 动态规划组件
   - 启发式剪枝

## 常见问题

### Q: 进化过程很慢怎么办？

A: 可以调整以下参数加速：
- 减少 `max_iterations`
- 减少测试用例数量（在evaluator.py中调整`load_test_cases()[:5]`）
- 增加 `parallel_evaluations`（如果有更多CPU核心）
- 使用更快的LLM模型

### Q: 所有解都无效怎么办？

A: 检查：
1. 初始程序能否正常运行
2. 评估器是否配置正确
3. 约束条件是否理解正确
4. LLM生成的代码是否有语法错误（查看日志）

### Q: 成本没有下降怎么办？

A: 尝试：
1. 增加 `temperature` 提高探索性
2. 增加 `population_size` 和 `num_islands`
3. 调整 `elite_selection_ratio` 和 `exploitation_ratio`
4. 修改 prompt 提供更多优化提示

### Q: 如何使用其他模型？

A: 修改 `config.qwen.yaml` 中的 `llm.primary_model`：

```yaml
# 使用 OpenAI GPT-4
llm:
  primary_model: "gpt-4"
  api_key: "your-openai-api-key"
  api_base: "https://api.openai.com/v1"

# 使用本地模型（需要兼容OpenAI API）
llm:
  primary_model: "local-model-name"
  api_base: "http://localhost:8000/v1"
```

## 进阶使用

### 可视化演化树

```bash
python ../../scripts/visualizer.py \
    --path openevolve_output/checkpoints/checkpoint_300/
```

### 分析最佳程序

```python
import json

# 读取最佳程序信息
with open('openevolve_output/best/best_program_info.json') as f:
    info = json.load(f)
    print(f"Generation: {info['generation']}")
    print(f"Found at iteration: {info['iteration']}")
    print(f"Metrics: {info['metrics']}")

# 读取并测试最佳程序
with open('openevolve_output/best/best_program.py') as f:
    best_code = f.read()
    print(best_code)
```

### 批量测试

```python
from evaluator import evaluate
import glob

for checkpoint_dir in sorted(glob.glob('openevolve_output/checkpoints/checkpoint_*')):
    program_path = f'{checkpoint_dir}/best_program.py'
    if os.path.exists(program_path):
        result = evaluate(program_path)
        print(f"{checkpoint_dir}: {result.metrics['combined_score']:.4f}")
```

## 参考资料

- [OpenEvolve 文档](../../CLAUDE.md)
- [华为 CodeCraft 2017 题目](README.md)
- [MAP-Elites 算法论文](https://arxiv.org/abs/1504.04909)
- [AlphaEvolve 论文](https://deepmind.google/discover/blog/alphaevolve/)

## 许可证

本示例遵循 OpenEvolve 项目的许可证。
