from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT, PRINCIPLE_MATCH_PROMPT
from inference.api_inference import gpt_call
from inference.local_inference import batch_inference, single_inference
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
import logging
import config
import json
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, path=None, similarity_threshold: float = None):
        # 如果环境变量未设置，则默认使用 0,1
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
        self.path = path or str(config.MEMORY_FILE)
        self.memory = self.load()
        # === 新增：加载嵌入模型 + 阈值 ===
        # 使用 cuda:0（因为 CUDA_VISIBLE_DEVICES 设置后，cuda:0 实际是物理GPU 0）
        self.embedder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, device='cuda:0')
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD

    def load(self):
        """加载内存数据，确保返回字典类型"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                return loaded_data
            else:
                print(f"警告：{self.path} 中数据不是字典类型，已初始化为空字典")
                return {}
        except FileNotFoundError:
            print(f"提示：{self.path} 不存在，已初始化为空字典")
            return {}
        except json.JSONDecodeError:
            print(f"错误：{self.path} 中JSON格式无效，已初始化为空字典")
            return {}

    def save(self):
        """保存内存字典到JSON文件"""
        if not isinstance(self.memory, dict):
            raise TypeError("self.memory 必须是字典类型，无法保存")

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve(self, task_desc: str):
        """先精确匹配 → 否则走语义相似度 ≥ 阈值"""
        normalized_task = task_desc.strip()

        # 1. 精确匹配
        # for task, principles in self.memory.items():
        #     if task.strip() == normalized_task:
        #         return task, principles

        # 2. 语义相似度匹配
        if not self.memory:
            return None, []

        input_vec = self.embedder.encode(normalized_task, convert_to_tensor=False)
        best_task, best_principles, best_score = None, [], -1

        for task, principles in self.memory.items():
            task_vec = self.embedder.encode(task, convert_to_tensor=False)
            score = self._cosine_similarity(input_vec, task_vec)
            if score > best_score:
                best_task, best_principles, best_score = task, principles, score

        if best_score >= self.similarity_threshold:
            return best_task, best_principles
        return None, []

    def add_task(self, task_desc: str, principles: list):
        self.memory[task_desc] = principles

    def merge_principles(self, task_desc: str, new_principles: list):
        old_principles = self.memory.get(task_desc, [])
        filtered = self._resolve_conflicts(old_principles, new_principles)
        self.memory[task_desc] = filtered

    def _resolve_conflicts(self, old: list, new: list) -> list:
        if not old:
            return new
        if not new:
            return old

        prompt = PRINCIPLE_MATCH_PROMPT.substitute(
            old="\n".join(old),
            new="\n".join(new)
        )
        result = single_inference(prompt)

        pattern = r'(\{\s*"comparisons"\s*:\s*\[.*?\]\s*\})'
        match = re.search(pattern, result, flags=re.DOTALL)
        if match:
            try:
                # 提取出来的字符串
                comparisons_json_str = match.group(1)
                # 加载为 Python 对象
                relations_dict = json.loads(comparisons_json_str)
                relations = relations_dict.get("comparisons", [])
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode failed: {e}\nRaw string:\n{comparisons_json_str}")
                relations = []
        else:
            print("[WARN] No valid 'comparisons' JSON found in model output:\n", result[:500])
            relations = []

        retained_old = set(old)
        retained_new = []

        # ✅ 安全检查，防止 NoneType 报错
        if not relations:
            print("[WARN] relations is empty, skipping conflict resolution.")
            return list(retained_old)

        for match in relations:
            old_rule = match.get("old", "").strip()
            new_rule = match.get("new", "").strip()
            relation = match.get("relation", "").strip()

            if relation == "Redundant":
                if old_rule in retained_old:
                    retained_old.remove(old_rule)
                retained_new.append(new_rule)
            elif relation == "Conflicting":
                if old_rule in retained_old:
                    retained_old.remove(old_rule)
                retained_new.append(new_rule)
            elif relation == "Irrelevant":
                retained_new.append(new_rule)

        return list(retained_old.union(retained_new))


