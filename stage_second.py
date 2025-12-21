from asyncio.log import logger
import json
import config
from pathlib import Path
from module.execute_module import batch_answer_questions_directly,concurrent_generate_chosen,batch_generate_task_descriptions
from module.plan_module import batch_generate_difference_list, batch_generate_principles
from module.memory_module import MemoryManager
import logging
import re
from tqdm import tqdm  
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def prepare_step2_update_memory_from_dpo():
    """
    使用DPO数据更新Memory
    """
    logger.info("=" * 60)
    logger.info("Step 2: 开始更新Memory")
    logger.info("=" * 60)
    
    # 创建 MemoryManager 对象
    memory = MemoryManager()
    logger.info("MemoryManager 初始化完成")
    
    dpo_file = Path(config.data_levels_file)
    checkpoint_file = Path(config.memory_checkpoint_file)
    
    if not dpo_file.exists():
        logger.error(f"DPO文件不存在: {dpo_file}，请先运行Step 1")
        return
    
    try:
        # 支持 JSONL 和 JSON 格式
        dpo_data = []
        if dpo_file.suffix == '.jsonl':
            with open(dpo_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dpo_data.append(json.loads(line))
        else:
            # 读取 LlamaFactory 格式的 JSON 文件（数组格式）
            with open(dpo_file, 'r', encoding='utf-8') as f:
                dpo_data = json.load(f)
        
        if not isinstance(dpo_data, list):
            logger.error(f"数据格式错误: 期望列表格式，实际为 {type(dpo_data)}")
            return
            
    except Exception as e:
        logger.error(f"读取JSON文件失败: {e}")
        return

    # 读取断点
    start_index = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as cf:
                ckpt = json.load(cf)
                start_index = ckpt.get("last_index", 0)
                logger.info(f"从断点恢复: {start_index}/{len(dpo_data)}")
        except Exception as e:
            logger.warning(f"读取断点文件失败: {e}，从头开始")
            start_index = 0

    # 主循环：批处理模式
    try:
        # ===== 阶段1: 数据预处理和过滤 =====
        logger.info("阶段1/3: 数据预处理和过滤")
        batch_data = []
        for i in range(start_index, len(dpo_data)):
            item = dpo_data[i]
            
            question = None
            diff = None
            
            # 1. 尝试从 input 字段提取 (LlamaFactory 格式)
            if 'input' in item and 'Error Analysis:' in item['input']:
                input_text = item['input']
                # 提取 Question
                # 匹配 "Question: " 开头，直到 "Error Analysis:" 前
                q_match = re.search(r'Question:\s*(.*?)\s*Error Analysis:', input_text, re.DOTALL)
                if q_match:
                    question = q_match.group(1).strip()
                
                # 提取 Diff
                # 匹配 "Error Analysis:" 后面的所有内容
                d_match = re.search(r'Error Analysis:\s*(.*?)$', input_text, re.DOTALL)
                if d_match:
                    diff = d_match.group(1).strip()
            
            # 2. 尝试从 messages 中提取 (JSONL 格式 - 兼容旧格式)
            elif 'messages' in item:
                user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
                if user_msg:
                    # 提取 Question 和 Diff
                    q_match = re.search(r'Input: Question: (.*?)\nError Points:', user_msg, re.DOTALL)
                    d_match = re.search(r'Error Points: (.*?)\nOutput:', user_msg, re.DOTALL)
                    
                    question = q_match.group(1).strip() if q_match else None
                    diff = d_match.group(1).strip() if d_match else None
            
            # 3. 尝试直接获取字段 (简单 JSON 格式)
            if not question:
                question = item.get("question")
            if not diff:
                diff = item.get("diff")

            if not question or not diff:
                logger.warning(f"第 {i} 项数据缺少question或diff，跳过")
                continue

            batch_data.append({
                'index': i,
                'question': question,
                'diff': diff
            })
        
        logger.info(f"准备批处理 {len(batch_data)} 条数据")
        
        # ===== 阶段2: 批量生成任务描述和原则 =====
        logger.info("阶段2/3: 批量生成任务描述和原则")
        
        # 批量生成任务描述
        questions = [item['question'] for item in batch_data]
        diffs = [item['diff'] for item in batch_data]
        
        logger.info("批量生成任务描述...")
        task_descs = batch_generate_task_descriptions(questions)
        
        # 批量生成新原则（使用弱模型）
        logger.info("批量生成新原则...")
        regenerated_list = batch_generate_principles(questions, diffs, model="weak")

        # === 新增：保存中间生成结果 ===
        intermediate_output_file = Path(config.output_dir) / "stage2_generated.json"
        logger.info(f"保存中间生成结果到: {intermediate_output_file}")
        intermediate_data = []
        for i, (q, d, td, rl) in enumerate(zip(questions, diffs, task_descs, regenerated_list)):
            intermediate_data.append({
                "index": batch_data[i]['index'],
                "question": q,
                "diff": d,
                "task_desc": td,
                "regenerated_principles": rl
            })
        
        # 确保输出目录存在
        intermediate_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(intermediate_output_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
        
        # ===== 阶段3: 解析结果并更新Memory =====
        logger.info("阶段3/3: 解析结果并更新Memory")
        
        for idx, item in enumerate(tqdm(batch_data, desc="更新Memory")):
            i = item['index']
            task_desc_new = task_descs[idx]
            regenerated = regenerated_list[idx]
            
            # 直接解析任务描述 JSON
            try:
                task_obj = json.loads(task_desc_new)
                desc = task_obj.get("taskDescription", {}).get("description")
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.warning(f"第 {i} 项解析任务描述失败: {e}，跳过")
                continue

            if not desc:
                logger.warning(f"第 {i} 项任务描述为空，跳过")
                continue

            # 获取已有 memory
            existing_key, existing_principles = memory.retrieve(desc)
            canonical_key = existing_key if existing_principles else desc

            # 直接解析新原则 JSON
            try:
                principles_obj = json.loads(regenerated)
                output_list = principles_obj.get("output", [])
                regenerated_parsed = [x.get("Principle") for x in output_list
                                      if isinstance(x, dict) and "Principle" in x]
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.warning(f"第 {i} 项解析原则失败: {e}")
                regenerated_parsed = None

            # 合并并保存
            if regenerated_parsed:
                memory.merge_principles(canonical_key, regenerated_parsed)
                memory.save()

            # 定期保存断点
            if (idx + 1) % config.SAVE_FREQUENCY == 0:
                with open(checkpoint_file, 'w') as cf:
                    json.dump({"last_index": i + 1}, cf)

        logger.info("Memory更新完成")
        
        # 任务结束清理断点
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            
    except Exception as e:
        logger.error(f"更新Memory时发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    主入口：支持直接运行此脚本
    """
    try:
        prepare_step2_update_memory_from_dpo()
        logger.info("程序执行完成")
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        raise