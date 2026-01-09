# S-Expression Generation Training 

## 🎯 概述

基于`sexpr_generation.py`的KBQA训练脚本，用于训练模型生成S-Expression而非直接生成SPARQL。

## 📂 文件说明

- `train_kbqa_sexpr_generation.sh` - 主训练脚本
- `setup_sexpr_training.sh` - 环境设置脚本  
- `../data_process/prepare_sexpr_training_data.py` - 数据准备脚本
- `../data_process/validate_sexpr_data.py` - 数据验证脚本

## 🚀 快速开始

### 1. 环境设置
```bash
./scripts/train/setup_sexpr_training.sh
```

### 2. 数据准备
```bash
# 处理WebQSP数据
python scripts/data_process/prepare_sexpr_training_data.py \
    --input_file /path/to/WebQSP.train.json \
    --dataset_type webqsp \
    --output_dir data/webqsp_sexpr \
    --split train

python scripts/data_process/prepare_sexpr_training_data.py \
    --input_file /path/to/WebQSP.test.json \
    --dataset_type webqsp \
    --output_dir data/webqsp_sexpr \
    --split test
```

### 3. 数据验证
```bash
python scripts/data_process/validate_sexpr_data.py \
    --data_dir data/webqsp_sexpr \
    --split train
```

### 4. 开始训练
```bash
# 8 GPU训练
./scripts/train/train_kbqa_sexpr_generation.sh

# 16 GPU训练  
./scripts/train/train_kbqa_sexpr_generation.sh ppu
```

## 📊 数据格式要求

训练数据需要包含以下字段：
- `prompt` - 包含问题和生成指令
- `response` - 包含推理过程和S-Expression
- `gold_sexpr` - 目标S-Expression
- `gold_answer` - 正确答案
- `question` - 原始问题

详细说明请参考 `docs/DATA_PREPARATION_SEXPR.md`

## ⚙️ 关键配置

### S-Expression特定配置
- `sexpr_config.enable_sexpr_mode=true` - 启用S-Expression模式
- `sexpr_config.enable_action_reasoning=true` - 启用动作推理
- `sexpr_config.enable_relation_retrieval=true` - 启用关系检索
- `sexpr_config.use_complete_sparql_converter=true` - 使用完整转换器

### 奖励模型配置
- `reward_model.sexpr_structure_score=0.2` - S-Expression结构奖励
- `reward_model.action_correctness_score=0.15` - 动作正确性奖励
- `reward_model.answer_accuracy_score=0.3` - 答案准确性奖励

## 🔧 自定义配置

修改训练脚本中的以下部分：
- `BASE_MODEL` - 模型路径
- `DATA_DIR` - 数据目录
- `EXPERIMENT_NAME` - 实验名称
- GPU和内存配置

## 📈 监控训练

训练日志将保存到 `$EXPERIMENT_NAME.log`，包含：
- S-Expression生成质量
- SPARQL转换成功率
- 答案准确性
- 奖励分数分布

## ❓ 常见问题

1. **数据转换失败**: 检查原始SPARQL格式是否标准
2. **内存不足**: 调整batch size或启用更多offloading
3. **转换器错误**: 确保S-Expression语法正确

详细故障排除请参考主文档。
