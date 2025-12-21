import os
from pathlib import Path
# 地址参数
Project_ROOT = Path(__file__).parent.absolute()#项目根路径，运行代码的父目录的绝对路径

check_root = os.path.join(Project_ROOT, 'checkpoints')  #模型检查点目录
output_dir = os.path.join(Project_ROOT, 'output')  #输出目录
data_dir = os.path.join(Project_ROOT, 'data')  #数据集目录
BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME', "/home/share/hcz/qwen2.5-14b-awq")
lora_model_path = os.getenv('LORA_MODEL_PATH', "/home/models/qwen_dpo2_lora")  #LoRA模型路径
MAX_MODEL_LEN = 4096
MEMORY_FILE =os.path.join(Project_ROOT, 'memory', 'memory.json')  #Memory文件路径

DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.9'))
DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P', '0.7'))
# 提升默认生成长度，避免接口截断；如需更短可通过环境变量覆盖
DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '8192'))

# ==================== Memory配置 ====================
 
SAVE_FREQUENCY = int(os.getenv('SAVE_FREQUENCY', '50'))  # 保存频率
# ==================== 日志配置 ====================
LOG_FILE = Project_ROOT / "logs"
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== 批处理和并发配置 ====================
# vLLM批处理大小（针对大数据集优化）
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))  # 平衡GPU利用率和内存占用

# API并发线程数
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '20'))  # 增加并发数
# Sentence Transformer模型路径
SENTENCE_TRANSFORMER_MODEL = os.path.join(Project_ROOT, 'em_model', 'all-MiniLM-L6-v2')  #本地Sentence Transformer模型路径
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))
original_data_file = os.path.join(data_dir, 'original_data', 'merged_all_levels.json')  #原始数据集文件路径
dpo_progress_file = os.path.join(check_root, 'dpo_progress.json')  #进度文件路径
memory_checkpoint_file = os.path.join(check_root, 'memory_progress.json')  #Memory断点文件路径
dpo_final_file = os.path.join(output_dir, 'dpo_final.jsonl')  #最终DPO数据文件路径
data_levels_file = os.path.join(data_dir, 'dpo_llamafactory', 'dpo_level_level2_llamafactory.json')  #数据级别文件路径

# 超参数
batch_size = 64  #本地推理批次大小
output_dir = os.path.join(Project_ROOT, 'output')  #输出目录

