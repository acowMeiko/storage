from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT
from inference.api_inference import gpt_call
from inference.local_inference import batch_inference, single_inference
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
import logging

logger = logging.getLogger(__name__)



# ==================== 单条推理函数（保持兼容） ====================
def concurrent_generate_chosen(questions: List[str], diffs: List[str], max_workers: int = 10) -> List[str]:
    """
    并发调用API生成Chosen原则（使用强模型API）
    
    Args:
        questions: 问题列表
        diffs: 差异分析列表
        max_workers: 最大并发数
    
    Returns:
        Chosen原则列表
    """
    logger.info(f"启动 {max_workers} 个并发线程调用强模型API")
    
    results = [None] * len(questions)
    
    def generate_single(idx: int, question: str, diff: str) -> tuple:
        """单个API调用"""
        try:
            chosen = generate_principles(question, diff, model="strong")
            return idx, chosen, None
        except Exception as e:
            logger.error(f"第 {idx} 项API调用失败: {e}")
            return idx, None, str(e)
    
    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(generate_single, idx, q, d): idx 
            for idx, (q, d) in enumerate(zip(questions, diffs))
        }
        
        # 收集结果（带进度条）
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="API并发调用"):
            idx, result, error = future.result()
            if error:
                logger.warning(f"第 {idx} 项使用默认值（API调用失败）")
                results[idx] = ""  # 或其他默认值
            else:
                results[idx] = result
    
    return results
# ==================== 批量推理函数（新增） ====================

def batch_generate_task_descriptions(questions: List[str]) -> List[str]:
    """
    批量生成任务描述
    
    Args:
        questions: 问题列表
        
    Returns:
        任务描述列表
    """
    from template.prompt_template import format
    # 先应用任务描述模板，再应用对话格式
    prompts = [
        format.substitute(query=TASK_DESC_PROMPT.substitute(question=q))
        for q in questions
    ]
    return batch_inference(prompts)
def generate_task_description(question: str, use_local: bool = True) -> str:
    """
    生成任务描述
    
    Args:
        question: 问题
        use_local: 是否使用本地模型（默认True）
    """
    prompt = TASK_DESC_PROMPT.substitute(question=question)
    if use_local:
        return single_inference(prompt)
    else:
        return gpt_call(user=prompt)

def answer_question_directly(question: str, use_local: bool = True) -> str:
    """
    直接回答问题
    
    Args:
        question: 问题
        use_local: 是否使用本地模型（默认True）
    """
    prompt = DIRECT_ANSWER_PROMPT.substitute(question=question)
    if use_local:
        return single_inference(prompt)
    else:
        return gpt_call(user=prompt)

def answer_with_principles(question: str, principles: list, use_local: bool = True) -> str:
    """
    根据原则回答问题
    
    Args:
        question: 问题
        principles: 原则列表
        use_local: 是否使用本地模型（默认True）
    """
    prompt = GUIDED_ANSWER_PROMPT.safe_substitute(
        question=question,
        principles="\n".join(principles) if principles else ""
    )
    if use_local:
        return single_inference(prompt)
    else:
        return gpt_call(user=prompt)


# ==================== 批量推理函数（新增） ====================

def batch_generate_task_descriptions(questions: List[str]) -> List[str]:
    prompts = [TASK_DESC_PROMPT.substitute(question=q) for q in questions]
    return batch_inference(prompts)


def batch_answer_questions_directly(questions: List[str]) -> List[str]:
    """
    批量直接回答问题
    
    Args:
        questions: 问题列表
        
    Returns:
        答案列表
    """
    from template.prompt_template import format
    prompts = [
        format.substitute(query=DIRECT_ANSWER_PROMPT.substitute(question=q))
        for q in questions
    ]
    return batch_inference(prompts)


def batch_answer_with_principles(questions: List[str], principles_list: List[list]) -> List[str]:
    """
    批量根据原则回答问题
    
    Args:
        questions: 问题列表
        principles_list: 原则列表（每个问题对应一个原则列表）
        
    Returns:
        答案列表
    """
    from template.prompt_template import format
    prompts = [
        format.substitute(query=GUIDED_ANSWER_PROMPT.safe_substitute(
            question=q,
            principles="\n".join(p) if p else ""
        ))
        for q, p in zip(questions, principles_list)
    ]
    return batch_inference(prompts)

