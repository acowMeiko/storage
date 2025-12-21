from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT, PRINCIPLE_PROMPT, DIFF_PROMPT
from inference.api_inference import gpt_call
from inference.local_inference import batch_inference, single_inference
from typing import List
import logging

logger = logging.getLogger(__name__)

# ==================== 单条推理函数（保持兼容） ====================

# CDE process.
def generate_difference_list(question: str, pred: str, label: str, use_local: bool = True) -> str:
    """
    生成差异列表
    
    Args:
        question: 问题
        pred: 预测答案
        label: 标准答案
        use_local: 是否使用本地模型（默认True）
    """
    prompt = DIFF_PROMPT.substitute(question=question, pred=pred, label=label)
    if use_local:
        from local_inference import single_inference
        return single_inference(prompt)
    else:
        return gpt_call(user=prompt)


def generate_principles(question: str, diff_list: str, model="weak", use_local: bool = True) -> str:
    """
    生成原则
    
    Args:
        question: 问题
        diff_list: 差异列表
        model: 模型类型 ("weak" 或 "strong")
        use_local: weak模型是否使用本地，strong始终使用API
    """
    prompt = PRINCIPLE_PROMPT.substitute(question=question, diff_list=diff_list)
    
    if model == "strong":
        # 强模型始终使用API
        return gpt_call(user=prompt, model=STRONG_MODEL_NAME, url=STRONG_MODEL_API_URL, api_key=STRONG_MODEL_KEY)
    else:
        # 弱模型根据参数决定
        if use_local:
            from local_inference import single_inference
            return single_inference(prompt)
        else:
            return gpt_call(user=prompt, model=BASE_MODEL_NAME)


# ==================== 批量推理函数（新增） ====================

def batch_generate_difference_list(questions: List[str], preds: List[str], labels: List[str]) -> List[str]:
    """
    批量生成差异列表
    
    Args:
        questions: 问题列表
        preds: 预测答案列表
        labels: 标准答案列表
        
    Returns:
        差异列表的列表
    """
    prompts = [
        DIFF_PROMPT.substitute(question=q, pred=p, label=l)
        for q, p, l in zip(questions, preds, labels)
    ]
    return batch_inference(prompts)


def batch_generate_principles(questions: List[str], diff_lists: List[str], model="weak") -> List[str]:
    """
    批量生成原则
    
    Args:
        questions: 问题列表
        diff_lists: 差异列表的列表
        model: 模型类型 ("weak" 使用本地, "strong" 使用API)
        
    Returns:
        原则列表
    """
    prompts = [
        PRINCIPLE_PROMPT.substitute(question=q, diff_list=d)
        for q, d in zip(questions, diff_lists)
    ]
    
    if model == "weak":
        # 弱模型使用本地批量推理
        return batch_inference(prompts)
    else:
        # 强模型使用API（在main_pipeline.py中并发调用）
        # 这里不应该被调用，如果需要应该使用concurrent_generate_chosen
        raise ValueError("强模型应使用concurrent_generate_chosen进行并发调用")

