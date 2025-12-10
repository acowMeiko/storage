"""
本地模型推理模块
支持vLLM批量推理和单条推理
"""
import logging
from typing import List, Optional, Union
from dataclasses import dataclass
import config
from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT,format
logger = logging.getLogger(__name__)

# 全局模型实例
_vllm_model = None
_sampling_params = None


@dataclass
class InferenceConfig:
    """推理配置"""
    temperature: float = config.DEFAULT_TEMPERATURE
    top_p: float = config.DEFAULT_TOP_P
    max_tokens: int = config.DEFAULT_MAX_TOKENS
    stop: Optional[List[str]] = None


def get_vllm_model():
    """
    获取或初始化vLLM模型（单例模式）
    
    Returns:
        vLLM模型实例
    """
    global _vllm_model, _sampling_params
    
    if _vllm_model is None:
        try:
            import os
            from vllm import LLM, SamplingParams
            
            # 如果环境变量未设置，则默认使用 0,1
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
                logger.info(f"环境变量 CUDA_VISIBLE_DEVICES 未设置，默认使用: {os.environ['CUDA_VISIBLE_DEVICES']}")
            else:
                logger.info(f"使用环境变量 CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            logger.info("=" * 60)
            logger.info("初始化vLLM本地模型...")
            logger.info(f"模型路径: {config.BASE_MODEL_NAME}")
            logger.info("=" * 60)
            
            _vllm_model = LLM(
                model=config.lora_model_path,
                tensor_parallel_size=2,  # 2张GPU并行
                gpu_memory_utilization=0.9,  # 80GB显存，可以使用更高利用率
                trust_remote_code=True,  # 信任远程代码（某些模型需要）
                dtype="auto",  # 自动选择数据类型
                max_model_len=config.MAX_MODEL_LEN,  # 使用配置的最大长度（默认8192）
            )
            
            _sampling_params = SamplingParams(
                temperature=config.DEFAULT_TEMPERATURE,
                top_p=config.DEFAULT_TOP_P,
                max_tokens=2048,  # 使用配置的最大token数
            )
            
            logger.info("vLLM模型初始化完成")
            
        except ImportError:
            logger.error("未安装vLLM，请运行: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"vLLM模型初始化失败: {e}", exc_info=True)
            raise
    
    return _vllm_model, _sampling_params


def batch_inference(
    prompts: List[str],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_tqdm: bool = True
) -> List[str]:
    """
    批量推理
    
    Args:
        prompts: 提示词列表
        temperature: 温度参数（可选）
        top_p: top_p参数（可选）
        max_tokens: 最大token数（可选）
        use_tqdm: 是否显示进度条
        
    Returns:
        生成的文本列表
    """
    if not prompts:
        return []
    
    model, default_params = get_vllm_model()
    
    # 创建采样参数
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature or config.DEFAULT_TEMPERATURE,
        top_p=top_p or config.DEFAULT_TOP_P,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
    )
    
    logger.debug(f"批量推理: {len(prompts)} 条数据")
    
    format_prompts = [format.substitute(query=q) for q in prompts]
    try:
        # vLLM批量生成
        outputs = model.generate(format_prompts, sampling_params, use_tqdm=use_tqdm)
        
        # 提取生成的文本
        results = [output.outputs[0].text.strip() for output in outputs]
        
        return results
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}", exc_info=True)
        raise


def single_inference(
    prompt: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    单条推理
    
    Args:
        prompt: 提示词
        temperature: 温度参数（可选）
        top_p: top_p参数（可选）
        max_tokens: 最大token数（可选）
        
    Returns:
        生成的文本
    """
    results = batch_inference(
        [prompt],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        use_tqdm=False
    )
    return results[0] if results else ""


def cleanup_model():
    """清理模型，释放显存"""
    global _vllm_model, _sampling_params
    
    if _vllm_model is not None:
        logger.info("清理vLLM模型...")
        del _vllm_model
        _vllm_model = None
        _sampling_params = None
        
        # 清理CUDA缓存
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("CUDA缓存已清理")
        except:
            pass


# 注册退出时清理
import atexit
atexit.register(cleanup_model)
