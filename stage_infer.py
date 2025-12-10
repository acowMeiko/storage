import json
import logging
import config
from pathlib import Path
from module.execute_module import generate_task_description, answer_with_principles
from module.memory_module import MemoryManager
from tqdm import tqdm
import re
import sys
import os

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def inference_with_memory(dataset):
    """
    使用Memory进行推理
    Args:
        dataset: 输入数据集
    """
    logger.info("=" * 60)
    logger.info("Step 3: 开始推理")
    logger.info("=" * 60)
    
    # Initialize MemoryManager
    try:
        memory = MemoryManager()
        logger.info("MemoryManager 初始化完成")
    except Exception as e:
        logger.error(f"MemoryManager 初始化失败: {e}")
        return

    results = []
    no_principles_count = 0
    
    # Define output file
    output_file = Path(config.output_dir) / "local_inference.json"
    
    try:
        for item in tqdm(dataset, desc="推理中"):
            question = item.get("question", "")
            if not question:
                logger.warning("数据项缺少question，跳过")
                continue
                
            try:
                # Generate task description
                task_desc_obj = generate_task_description(question)
                # Clean up potential think tags if model outputs them
                task_cleaned = re.sub(r"<think>.*?</think>", "", task_desc_obj, flags=re.DOTALL).strip()
                
                # Extract JSON
                # Try to find JSON block if not pure JSON
                json_match = re.search(r'\{.*\}', task_cleaned, re.DOTALL)
                if json_match:
                    task_cleaned = json_match.group(0)
                
                parsed = json.loads(task_cleaned)
                task_desc = parsed["taskDescription"]["description"]
            except Exception as e:
                logger.warning(f"解析任务描述失败: {e}，跳过该项")
                continue
                
            # Retrieve principles from memory
            task_key, principles = memory.retrieve(task_desc)
            
            if not principles:
                logger.debug(f"未找到原则: {task_desc}")
                no_principles_count += 1
                continue

            # Answer with principles
            answer = answer_with_principles(question, principles)
            
            results.append({
                "question": question,
                "task": task_desc,
                "answer": answer,
                "principles_used": principles
            })

            # Save periodically
            if len(results) % 10 == 0:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        # Final save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"推理完成，结果保存到: {output_file}")
        logger.info(f"成功推理: {len(results)} 项")
        logger.info(f"未找到原则: {no_principles_count} 项")
        
        # update_rejected_responses() # Removed as it is undefined
        
    except Exception as e:
        logger.error(f"推理过程发生错误: {e}", exc_info=True)
        raise

def main():
    # Load dataset
    input_file = Path(config.original_data_file)
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return

    try:
        logger.info(f"加载数据集: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"数据集加载成功，共 {len(dataset)} 条数据")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return

    # Run inference
    inference_with_memory(dataset)

if __name__ == "__main__":
    main()