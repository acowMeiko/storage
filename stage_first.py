from asyncio.log import logger
import json
import config
from pathlib import Path
from module.execute_module import batch_answer_questions_directly,concurrent_generate_chosen
from module.plan_module import batch_generate_difference_list, batch_generate_principles
import logging  
import os
import tqdm
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def batch_answer_questions(questions: List[str]) -> List[str]:
    """
    批量生成Baseline答案（使用本地vLLM）
    
    Args:
        questions: 问题列表
    
    Returns:
        答案列表
    """
    logger.info(f"批量生成Baseline答案: {len(questions)} 条")
    return batch_answer_questions_directly(questions)


def batch_generate_differences(questions: List[str], preds: List[str], labels: List[str]) -> List[str]:
    """
    批量生成差异分析（使用本地vLLM）
    
    Args:
        questions: 问题列表
        preds: 预测答案列表
        labels: 标准答案列表
    
    Returns:
        差异分析列表
    """
    logger.info(f"批量生成差异分析: {len(questions)} 条")
    return batch_generate_difference_list(questions, preds, labels)


def batch_generate_principles_local(questions: List[str], diffs: List[str], model: str = "weak") -> List[str]:
    """
    批量生成原则（使用本地vLLM）
    
    Args:
        questions: 问题列表
        diffs: 差异分析列表
        model: 模型类型（"weak" 使用本地模型）
    
    Returns:
        原则列表
    """
    logger.info(f"批量生成原则（弱模型）: {len(questions)} 条")
    return batch_generate_principles(questions, diffs, model=model)
def prepare_stage1(dataset):
    """
    生成DPO训练数据
    Args:
        dataset: 输入数据集
    """
    logger.info("=" * 60)
    logger.info("Step 1: 开始生成DPO数据")
    logger.info("=" * 60)
    progress_file = Path(config.dpo_progress_file)
    dpo_file = Path(config.dpo_final_file)
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            start_idx = progress.get('processed_count', 0)
            logger.info(f"发现已有进度，将从第 {start_idx} 项开始继续处理")
        except Exception as e:
            logger.error(f"读取进度文件失败: {e}，将从头开始")
            start_idx = 0
    else:
        start_idx = 0
        logger.info("未发现已有进度，将从头开始处理")
        # 如果重新开始，清空现有JSONL文件
        if dpo_file.exists():
            dpo_file.unlink()
    # 从上次中断的位置继续处理
    try:
        # 准备所有待处理数据
        all_data = []
        for i in range(start_idx, len(dataset)):
            item = dataset[i]
            # 兼容 problem/question 字段
            question = item.get("problem") or item.get("question", "")
            label = item.get("answer", "")
            
            if not question or not label:
                logger.warning(f"第 {i} 项数据缺少problem/question或answer，跳过")
                continue
            
            all_data.append({
                'index': i,
                'question': question,
                'label': label
            })
        total_items = len(all_data)
        logger.info(f"共需处理 {total_items} 条数据，批次大小: {config.batch_size}")
        
        
        logger.info("=" * 60)
        logger.info("阶段1/3: vLLM分批本地推理（所有数据）")
        logger.info("=" * 60)

        all_baseline_answers = []
        all_diffs = []
        all_rejected = []
        for batch_start in range(0, total_items, config.BATCH_SIZE):
            batch_end = min(batch_start + config.BATCH_SIZE, total_items)
            batch_data = all_data[batch_start:batch_end]
            
            logger.info(f"处理批次 [{batch_start+1}-{batch_end}/{total_items}]")
            
            # 批量获取问题列表
            questions = [item['question'] for item in batch_data]
            labels = [item['label'] for item in batch_data]
            
            # 批量推理：Baseline answers
            logger.info(f"  → 生成Baseline答案 ({len(questions)} 条)...")
            baseline_answers = batch_answer_questions(questions)
            all_baseline_answers.extend(baseline_answers)
            
            # 批量推理：Difference analysis
            logger.info(f"  → 生成差异分析 ({len(questions)} 条)...")
            diffs = batch_generate_differences(questions, baseline_answers, labels)
            all_diffs.extend(diffs)
            
            # 批量推理：Rejected responses
            logger.info(f"  → 生成Rejected原则 ({len(questions)} 条)...")
            rejected_list = batch_generate_principles_local(questions, diffs, model="weak")
            all_rejected.extend(rejected_list)
            
            logger.info(f"批次 [{batch_start+1}-{batch_end}] 本地推理完成")
        
        logger.info(f"阶段1完成: 共生成 {len(all_baseline_answers)} 条本地推理结果")
    except Exception as e:
        logger.error(f"生成DPO数据时发生错误: {e}", exc_info=True)
        raise
    vllm_cache_file = OUTPUTS_DIR / "vllm_cache.json"
    logger.info(f"保存vLLM处理结果到: {vllm_cache_file}")
    vllm_cache_data = {
        'all_data': all_data,
        'all_baseline_answers': all_baseline_answers,
        'all_diffs': all_diffs,
        'all_rejected': all_rejected,
        'total_items': total_items
    }
    vllm_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vllm_cache_file, 'w', encoding='utf-8') as f:
        json.dump(vllm_cache_data, f, indent=2, ensure_ascii=False)
    logger.info("vLLM处理结果已安全保存")
           # ===== 阶段2: 并发调用API处理所有数据（分批） =====
    logger.info("=" * 60)
    logger.info("阶段2/3: API并发生成Chosen（分批处理）")
    logger.info("=" * 60)
    
    all_questions = [item['question'] for item in all_data]
    all_chosen = []
    
    # 计算API批次大小（避免一次性并发过多）
    api_batch_size = config.MAX_WORKERS # 例如15个worker，每批1500条
    total_api_batches = (len(all_questions) + api_batch_size - 1) // api_batch_size
    
    logger.info(f"API分批处理: 每批 {api_batch_size} 条，共 {total_api_batches} 批")
    
    for api_batch_idx in range(0, len(all_questions), api_batch_size):
        api_batch_end = min(api_batch_idx + api_batch_size, len(all_questions))
        batch_questions = all_questions[api_batch_idx:api_batch_end]
        batch_diffs = all_diffs[api_batch_idx:api_batch_end]
        
        logger.info(f"API批次 [{api_batch_idx+1}-{api_batch_end}/{len(all_questions)}] 开始处理...")
        batch_chosen = concurrent_generate_chosen(batch_questions, batch_diffs, max_workers=config.MAX_WORKERS)
        all_chosen.extend(batch_chosen)
        
        logger.info(f"API批次 [{api_batch_idx+1}-{api_batch_end}] 完成")
    
    logger.info(f"阶段2完成: 共生成 {len(all_chosen)} 条Chosen结果")
    
    # ===== 阶段3: 组装所有DPO数据并保存为JSONL =====
    logger.info("=" * 60)
    logger.info("阶段3/3: 组装DPO数据并保存为JSONL格式")
    logger.info("=" * 60)
    
    # 确保输出目录存在
    dpo_file.parent.mkdir(parents=True, exist_ok=True)
    # 打开文件以追加模式（如果是续传）或写入模式（如果是新开始）
    file_mode = 'a' if start_idx > 0 else 'w'
    
    saved_count = 0
    with open(dpo_file, file_mode, encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(all_data, desc="组装并保存JSONL")):
            i = item['index']
            question = item['question']
            diff = all_diffs[idx]
            rejected = all_rejected[idx]
            chosen = all_chosen[idx]
            
            # 构建符合DPO训练格式的数据
            dpo_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates reusable problem-solving principles."
                    },
                    {
                        "role": "user",
                        "content": f"Input: Question: {question}\nError Points: {diff}\nOutput: Reusable principles"
                    },
                    {
                        "role": "assistant",
                        "content": chosen
                    }
                ],
                "rejected_response": rejected
            }
            
            # 写入JSONL格式（每行一个JSON对象）
            f.write(json.dumps(dpo_item, ensure_ascii=False) + '\n')
            saved_count += 1
            
            # 定期刷新缓冲区并保存进度
            if saved_count % config.SAVE_FREQUENCY == 0:
                f.flush()  # 确保数据写入磁盘
                logger.info(f"已保存 {saved_count}/{total_items} 条到JSONL")
                
                # 保存进度
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, 'w', encoding='utf-8') as pf:
                    json.dump({
                        'processed_count': i + 1,
                        'total_count': len(dataset),
                        'last_processed_index': i
                    }, pf, indent=2)

    logger.info(f"DPO数据生成完成: {dpo_file}")
    logger.info(f"共保存 {saved_count} 条数据到JSONL格式")
        
    # 处理完成后删除进度文件
    if progress_file.exists():
        progress_file.unlink()
    

def main():
    input_file = Path(config.original_data_file)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        print(f"[✘] 错误: 输入文件不存在 - {input_file}")
        return
    try:
        logger.info(f"加载数据集: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"数据集加载成功，共 {len(dataset)} 条数据")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}", exc_info=True)
        print(f"[✘] 错误: 无法加载数据集 - {e}")
        return
    prepare_stage1(dataset)
if __name__ == "__main__":
    main()