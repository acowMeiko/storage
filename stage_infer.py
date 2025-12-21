import json
import logging
import config
from pathlib import Path
from module.execute_module import (
    batch_generate_task_descriptions, 
    batch_answer_with_principles
)
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

def extract_answer(text):
    """
    从模型输出中提取答案
    支持多种格式：\\boxed{}, **Answer:**, 最终答案、#### 等
    """
    if not text:
        return ""
    
    # 1. 尝试提取 #### 后面的答案（最优先）
    hash_answer_match = re.search(r'####\s*(.+?)(?:\n|$)', text, re.DOTALL)
    if hash_answer_match:
        answer = hash_answer_match.group(1).strip()
        # 清理可能的标签
        answer = re.sub(r'<[^>]+>', '', answer).strip()
        if answer and answer not in ['null', 'NULL']:
            return answer
    
    # 2. 尝试提取 \\boxed{} 格式
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 3. 尝试提取 **Answer:** 或 **答案:** 格式
    answer_match = re.search(r'\*\*(?:Answer|答案)[:：]\*\*\s*([^\n]+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 4. 尝试提取 "最终答案" 或 "Final Answer"
    final_match = re.search(r'(?:最终答案|Final Answer)[:：]\s*([^\n]+)', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # 5. 尝试提取最后一行非空内容作为答案
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 过滤掉系统标签
    valid_lines = [line for line in lines 
                   if not re.match(r'^<[^>]+>$', line)]  # 过滤纯标签行
    if valid_lines:
        return valid_lines[-1]
    
    return text.strip()

def normalize_answer(answer):
    """
    标准化答案格式，用于比较
    - 去除空格
    - 统一大小写
    - 去除标点符号
    """
    if not answer:
        return ""
    
    # 转为字符串
    answer = str(answer)
    
    # 去除LaTeX命令但保留内容
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', answer)
    
    # 去除多余空格
    answer = re.sub(r'\s+', '', answer)
    
    # 转小写
    answer = answer.lower()
    
    # 去除常见标点
    answer = re.sub(r'[,，.。;；:：!！?？]', '', answer)
    
    return answer

def check_answer_correctness(predicted, ground_truth):
    """
    检查答案是否正确
    返回: (is_correct: bool, normalized_pred: str, normalized_gt: str)
    """
    # 提取预测答案
    extracted_pred = extract_answer(predicted)
    
    # 标准化
    norm_pred = normalize_answer(extracted_pred)
    norm_gt = normalize_answer(ground_truth)
    
    # 完全匹配
    is_correct = norm_pred == norm_gt
    
    return is_correct, extracted_pred, norm_gt

def calculate_accuracy(results):
    """
    计算整体准确率
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0
        }
    
    correct_count = 0
    
    for result in results:
        ground_truth = result.get("ground_truth_answer", "")
        predicted = result.get("answer", "")
        
        is_correct, extracted_pred, norm_gt = check_answer_correctness(predicted, ground_truth)
        
        # 更新结果中的检查信息
        result["is_correct"] = is_correct
        result["extracted_answer"] = extracted_pred
        
        if is_correct:
            correct_count += 1
    
    # 计算准确率
    accuracy = {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0
    }
    
    return accuracy

def inference_with_memory(dataset):
    """
    使用Memory进行推理（批处理版本）
    Args:
        dataset: 输入数据集
    """
    logger.info("=" * 60)
    logger.info("Step 3: 开始推理（批处理模式）")
    logger.info("=" * 60)
    
    # Initialize MemoryManager
    try:
        memory = MemoryManager()
        logger.info("MemoryManager 初始化完成")
    except Exception as e:
        logger.error(f"MemoryManager 初始化失败: {e}")
        return

    results = []
    output_file = Path(config.output_dir) / "local_inference.json"
    
    # 批处理大小
    batch_size = config.BATCH_SIZE
    logger.info(f"批处理大小: {batch_size}")
    
    try:
        # 第一步：批量生成任务描述
        logger.info("=" * 60)
        logger.info("步骤 1/3: 批量生成任务描述")
        logger.info("=" * 60)
        
        valid_items = []
        questions = []
        
        for item in dataset:
            question = item.get("problem", "")
            if question:
                valid_items.append(item)
                questions.append(question)
            else:
                logger.warning("数据项缺少problem，跳过")
        
        logger.info(f"有效数据: {len(valid_items)} 条")
        
        # 批量生成任务描述
        all_task_descs = []
        num_batches = (len(questions) + batch_size - 1) // batch_size
        logger.info(f"将分 {num_batches} 批处理（每批 {batch_size} 条）")
        
        for i in tqdm(range(0, len(questions), batch_size), desc="生成任务描述", total=num_batches):
            batch_questions = questions[i:i+batch_size]
            batch_task_descs = batch_generate_task_descriptions(batch_questions)
            all_task_descs.extend(batch_task_descs)
        
        logger.info(f"任务描述生成完成: {len(all_task_descs)} 条")
        
        # 保存任务描述到JSON文件（中间结果）
        task_desc_file = Path(config.output_dir) / "task_descriptions.json"
        task_desc_file.parent.mkdir(parents=True, exist_ok=True)
        task_desc_data = [
            {
                "index": idx,
                "problem": item.get("problem"),
                "task_description_raw": task_desc
            }
            for idx, (item, task_desc) in enumerate(zip(valid_items, all_task_descs))
        ]
        with open(task_desc_file, 'w', encoding='utf-8') as f:
            json.dump(task_desc_data, f, indent=2, ensure_ascii=False)
        logger.info(f"任务描述已保存到: {task_desc_file}")
        
        # 第二步：解析任务描述并检索原则
        logger.info("=" * 60)
        logger.info("步骤 2/3: 解析任务描述并检索原则")
        logger.info("=" * 60)
        
        items_with_principles = []
        questions_for_answer = []
        principles_for_answer = []
        
        parse_failed_count = 0
        empty_response_count = 0
        
        for idx, (item, task_desc_obj) in enumerate(tqdm(zip(valid_items, all_task_descs), 
                                         total=len(valid_items),
                                         desc="检索原则")):
            try:
                # 检查是否为空响应
                if not task_desc_obj or not task_desc_obj.strip():
                    empty_response_count += 1
                    if empty_response_count <= 3:  # 只显示前3个
                        logger.warning(f"第 {idx+1} 项: 模型返回空响应")
                    continue
                
                # 策略：找到所有可能的JSON对象，取最后一个有效的
                # 使用更宽松的正则，匹配包含taskDescription的JSON
                json_candidates = []
                
                # 方法1：查找所有大括号对，尝试解析包含taskDescription的
                brace_count = 0
                start_pos = -1
                for i, char in enumerate(task_desc_obj):
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos != -1:
                            potential_json = task_desc_obj[start_pos:i+1]
                            # 只保留包含taskDescription的JSON
                            if 'taskDescription' in potential_json and 'description' in potential_json:
                                json_candidates.append(potential_json)
                
                # 如果没找到，尝试简单的正则匹配
                if not json_candidates:
                    json_match = re.search(r'\{.*?"taskDescription".*?\}', task_desc_obj, re.DOTALL)
                    if json_match:
                        json_candidates.append(json_match.group(0))
                
                if not json_candidates:
                    parse_failed_count += 1
                    if parse_failed_count <= 3:
                        logger.warning(f"第 {idx+1} 项: 未找到JSON格式，原始响应: {task_desc_obj[:200]}...")
                    continue
                
                # 从后往前尝试解析，取第一个成功的（即最后一个有效的）
                parsed = None
                json_str = None
                for candidate in reversed(json_candidates):
                    try:
                        parsed = json.loads(candidate)
                        json_str = candidate
                        break
                    except json.JSONDecodeError:
                        continue
                
                if parsed is None:
                    parse_failed_count += 1
                    if parse_failed_count <= 3:
                        logger.warning(f"第 {idx+1} 项: JSON解析失败")
                        logger.warning(f"  候选JSON数量: {len(json_candidates)}")
                        logger.warning(f"  最后一个候选: {json_candidates[-1][:150]}...")
                    continue
                
                # 检查必需字段
                if "taskDescription" not in parsed or "description" not in parsed["taskDescription"]:
                    parse_failed_count += 1
                    if parse_failed_count <= 3:
                        logger.warning(f"第 {idx+1} 项: JSON缺少必需字段，内容: {json_str[:100]}...")
                    continue
                    
                task_desc = parsed["taskDescription"]["description"]
                
                # 从Memory检索原则
                task_key, principles = memory.retrieve(task_desc)
                
                # 无论是否找到原则，都加入处理队列
                items_with_principles.append(item)
                questions_for_answer.append(item.get("problem"))
                
                if principles:
                    # 找到原则，使用原则指导
                    principles_for_answer.append(principles)
                    item["_task_desc"] = task_desc
                    item["_has_principles"] = True
                else:
                    # 未找到原则，使用空列表（模型将直接回答）
                    principles_for_answer.append([])
                    item["_task_desc"] = task_desc
                    item["_has_principles"] = False
                    logger.debug(f"未找到原则，将使用直接回答模式: {task_desc}")
                    
            except json.JSONDecodeError as e:
                parse_failed_count += 1
                if parse_failed_count <= 3:  # 只显示前3个解析错误
                    logger.warning(f"第 {idx+1} 项: JSON解析失败 - {e}")
                    logger.warning(f"  提取的JSON: {json_str[:150] if 'json_str' in locals() else 'N/A'}...")
                    logger.warning(f"  原始响应: {task_desc_obj[:150]}...")
                continue
            except Exception as e:
                parse_failed_count += 1
                if parse_failed_count <= 3:
                    logger.warning(f"第 {idx+1} 项: 处理失败 - {e}")
                continue
        
        logger.info(f"成功解析任务描述: {len(items_with_principles)} 条")
        logger.info(f"  - 找到原则: {sum(1 for item in items_with_principles if item.get('_has_principles', False))} 条")
        logger.info(f"  - 未找到原则（将直接回答）: {sum(1 for item in items_with_principles if not item.get('_has_principles', False))} 条")
        logger.info(f"解析失败: {len(valid_items) - len(items_with_principles)} 条")
        logger.info(f"  - 空响应: {empty_response_count} 条")
        logger.info(f"  - JSON解析失败: {parse_failed_count} 条")
        
        # 如果所有任务描述都失败，给出警告
        if len(items_with_principles) == 0 and len(valid_items) > 0:
            logger.warning("\n" + "="*60)
            logger.warning("警告: 所有任务描述生成都失败了！")
            logger.warning("可能原因:")
            logger.warning("  1. 模型未正确加载或配置")
            logger.warning("  2. TASK_DESC_PROMPT 模板有问题")
            logger.warning("  3. 模型输出格式不符合预期")
            logger.warning("建议: 运行 python check_task_generation.py 检查任务描述生成")
            logger.warning("="*60 + "\n")
        
        # 第三步：批量生成答案
        logger.info("=" * 60)
        logger.info("步骤 3/3: 批量生成答案")
        logger.info("=" * 60)
        
        num_answer_batches = (len(questions_for_answer) + batch_size - 1) // batch_size
        logger.info(f"将分 {num_answer_batches} 批处理（每批 {batch_size} 条）")
        
        all_answers = []
        for i in tqdm(range(0, len(questions_for_answer), batch_size), desc="生成答案", total=num_answer_batches):
            batch_questions = questions_for_answer[i:i+batch_size]
            batch_principles = principles_for_answer[i:i+batch_size]
            batch_answers = batch_answer_with_principles(batch_questions, batch_principles)
            all_answers.extend(batch_answers)
        
        logger.info(f"答案生成完成: {len(all_answers)} 条")
        
        # 组装最终结果
        logger.info("=" * 60)
        logger.info("组装结果")
        logger.info("=" * 60)
        
        for item, answer, principles in zip(items_with_principles, all_answers, principles_for_answer):
            result_item = {
                "problem": item.get("problem"),
                "task": item.get("_task_desc"),
                "answer": answer,
                "has_principles": item.get("_has_principles", False),
                "principles_used": principles if principles else [],
                "ground_truth_answer": item.get("answer"),
            }
            
            # 保留原始数据的可选字段（兼容多种格式）
            optional_fields = ["subject", "level", "difficulty", "tags", "unique_id", "solution"]
            for field in optional_fields:
                if field in item:
                    result_item[field] = item.get(field)
            
            results.append(result_item)
        
        # 保存结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 计算准确率
        logger.info("=" * 60)
        logger.info("开始计算准确率")
        logger.info("=" * 60)
        
        accuracy_stats = calculate_accuracy(results)
        
        # 保存带准确率标注的结果
        output_file_with_check = Path(config.output_dir) / "local_inference_with_accuracy.json"
        with open(output_file_with_check, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存准确率统计
        accuracy_file = Path(config.output_dir) / "accuracy_stats.json"
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            json.dump(accuracy_stats, f, indent=2, ensure_ascii=False)
        
        # 打印准确率报告
        logger.info(f"\n{'='*60}")
        logger.info("准确率报告")
        logger.info(f"{'='*60}")
        if accuracy_stats.get('total', 0) > 0:
            logger.info(f"总体准确率: {accuracy_stats['correct']}/{accuracy_stats['total']} = {accuracy_stats['accuracy']:.2%}")
        else:
            logger.info("无有效结果，无法计算准确率")
        
        logger.info(f"\n结果已保存:")
        logger.info(f"  - 推理结果: {output_file}")
        logger.info(f"  - 带准确率标注: {output_file_with_check}")
        logger.info(f"  - 准确率统计: {accuracy_file}")
            
        logger.info(f"\n推理完成!")
        logger.info(f"  - 成功推理: {len(results)} 项")
        logger.info(f"  - 未找到原则: {len(valid_items) - len(items_with_principles)} 项")
        logger.info(f"  - 总数据量: {len(dataset)} 项")
        
    except Exception as e:
        logger.error(f"推理过程发生错误: {e}", exc_info=True)
        raise

def main():
    # Load dataset - JSON格式 (test_filter.json)
    # 可以通过命令行参数指定，默认使用 test_filter.json
    import sys
    input_file = Path(os.getenv('TEST_FILE', 'data/test/test_filter.json'))
    try:
        logger.info(f"加载数据集: {input_file}")
        
        # JSON格式：整个文件是一个JSON数组
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"数据集加载成功，共 {len(dataset)} 条数据")
        
        # 显示数据格式信息
        if dataset:
            sample = dataset[0]
            logger.info(f"数据字段: {list(sample.keys())}")
            
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return

    # Run inference
    inference_with_memory(dataset)

if __name__ == "__main__":
    main()
