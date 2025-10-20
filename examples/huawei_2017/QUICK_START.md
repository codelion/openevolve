# 快速开始 - 华为2017 CDN优化

## 5分钟快速运行

### 1. 确认环境
```bash
cd openevolve/examples/huawei_2017
python -c "from evaluator import evaluate; print('OK')"
```

### 2. 测试初始程序
```bash
python initial_program.py case_example/batch1/0Primary/case0.txt
```

应该看到成本输出和路径方案。

### 3. 验证评估
```bash
python -c "from evaluator import evaluate; r = evaluate('initial_program.py'); print(f'Score: {r.metrics[\"combined_score\"]:.4f}')"
```

应该看到 Score: ~0.6267

### 4. 运行进化（短测试）
```bash
python ..\..\openevolve-run.py initial_program.py evaluator.py --config config.qwen.yaml --iterations 100 --checkpoint .\openevolve_output\checkpoints\checkpoint_975
```

### 5. 查看结果
```bash
cat openevolve_output/best/best_program_info.json
```

## 完整运行（300次迭代）

**Windows**:
```batch
run_evolution.bat
```

**Linux/Mac**:
```bash
chmod +x run_evolution.sh
./run_evolution.sh
```

预计时间: 3-5小时（取决于API速度）

## 关键文件

- ✅ `initial_program.py` - 基线程序（成本~150,000）
- ✅ `evaluator.py` - 评估器（测试5个案例）
- ✅ `config.qwen.yaml` - Qwen模型配置
- 📁 `case_example/` - 测试用例（80个）
- 📊 `openevolve_output/` - 进化结果（运行后生成）

## 预期效果

| 指标 | 初始值 | 目标 |
|------|--------|------|
| 有效率 | 100% | 100% |
| 平均成本 | 150,000 | < 100,000 |
| 综合得分 | 0.63 | > 0.75 |

## 常见问题

### API Key配置

编辑 `config.qwen.yaml`:
```yaml
llm:
  api_key: "your-api-key-here"
  api_base: "https://your-api-endpoint"
```

### 加速测试

减少迭代次数和测试用例：
```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
    --config config.qwen.yaml --iterations 50
```

在 `evaluator.py` 中修改:
```python
test_cases = load_test_cases(case_dir)[:2]  # 只测试2个案例
```

### 查看详细日志

```bash
tail -f openevolve_output/logs/openevolve_*.log
```

## 下一步

详细文档请查看:
- `GUIDE.md` - 完整使用指南
- `README.md` - 问题描述
- `../../CLAUDE.md` - OpenEvolve架构

祝优化顺利！ 🚀
