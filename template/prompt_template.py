"""Prompt Templates and Model Configuration"""
from string import Template
import config

# 从 config 导入配置，保持向后兼容
# BASE_MODEL_API_URL = config.BASE_MODEL_API_URL
BASE_MODEL_NAME = config.BASE_MODEL_NAME

# STRONG_MODEL_API_URL = config.STRONG_MODEL_API_URL
# STRONG_MODEL_NAME = config.STRONG_MODEL_NAME
# STRONG_MODEL_KEY = config.STRONG_MODEL_KEY

DIRECT_ANSWER_PROMPT = Template('''
<system>
You are an expert in solving problems.Please answer the question below. Let's think step by step.
</system>
$question
''')

from string import Template

GUIDED_ANSWER_PROMPT = Template(r'''
<system>
You are an expert in solving problems. Provide a clear step-by-step solution.
After your step-by-step reasoning, you MUST output the final answer in EXACTLY this format:

#### <final_answer>

Rules for <final_answer>:
- For numeric answers: output the number directly (e.g., #### 42 or #### 3.14 or #### -5)
- For expressions: output the expression (e.g., #### \\frac{3}{4} or #### 2\\pi)
- For text answers: output the text (e.g., #### Yes or #### \\text{Evelyn})
- The answer should be on a single line after ####
- No extra text, quotes, or tags after the #### line

Examples:
#### 14400
#### \\frac{14}{3}
#### 3.1415
#### \\text{Yes}
#### -2

Now solve the question below step by step, and finish by outputting the final answer in the format specified above.
</system>

Please ensure your response adheres to the specified principles.

<input>
Question: ${question}
Principles to follow:
${principles}
</input>
''')

# EXTRSCT_ANWSER_PROMPT = Template('''
# <system>
# You are an expert in extracting answers from long text. Please extract the answer to the question from the given text.
# </system>
# <input>
# Question: $question
# Text: $text
# </input>
# ''')
format = Template("""<|im_start|>user\n$query<|im_end|>\n<|im_start|>assistant\n""")

TASK_DESC_PROMPT = Template('''
Your task is to identify and extract the main task description from a given question. First, analyze the domain of the question, categorize it into a relevant subcategory, and then generate a concise, clear, and abstract task description that reflects the core objective.

**Steps to Perform Structured Analysis:**
1. **Analyze the domain of the question:** Determine the field or category the question belongs to
2. **Categorize the task:** Identify the specific type of problem within that domain
3. **Generate the task description:** Based on the identified domain and subcategory, create a task description that is:
   - Concise and clear
   - Abstract and general
   - Focused on the core objective
   - Free of unnecessary details or background information

**Question:** 
$question

**Output Format (JSON only, no extra text):**
{{
  "taskDescription": {{
    "description": "Clear, abstract, and specific description of the task, focusing on the core action or objective."
  }}
}}
''')

DIFF_PROMPT = Template('''
You are an expert in analyzing and comparing task responses to identify *fine-grained*, *task-relevant*, and *impactful* differences between answers that affect quality.

**Task:**
Given a high-quality and a low-quality answer to the same task, identify detailed differences that reflect meaningful changes in correctness, reasoning, completeness, or clarity.

**Steps:**
1. **Understand the task type:** Identify whether the task involves reasoning, generation, factual recall, explanation, etc.
2. **Perform targeted comparison:** Compare the answers component by component (sentence by sentence, step by step, or idea by idea).
3. **Identify key differences:** For each meaningful difference:
   - Quote or paraphrase the specific content from both answers
   - Indicate the aspect being affected
   - Explain why this difference matters

**Important guidelines:**
- Avoid vague language like "clearer" or "more logical" unless supported by concrete details
- Specify missing steps, incorrect reasoning, unsupported claims, or structural flaws
- Use task-specific language

**Input:**
Question: $question
Low-quality Answer: $pred
High-quality Answer: $label

**Output (JSON only, no extra text):**
{{
  "differences": [
    {{
      "Aspect": "Aspect being evaluated",
      "HighQuality": "Quoted or paraphrased content from the HQ answer",
      "LowQuality": "Quoted or paraphrased content from the LQ answer",
      "Differences": "Detailed explanation of why this difference affects answer quality"
    }}
  ]
}}
''')
 
PRINCIPLE_PROMPT = Template('''
You are a prompt engineering expert skilled in deriving precise and generalizable principles that improve language model outputs. Your task is to formulate principles based on observed differences between high- and low-quality answers, ensuring each principle reflects a specific failure pattern and offers guidance for correction.

**Task:**
Generate reusable and insightful improvement principles based on observed differences between two answers.

**Steps:**
1. Carefully examine each identified difference and explain how it impacts the answer quality (e.g., logical structure, factual accuracy, clarity, completeness, or relevance).
2. For each difference, derive a principle that captures the core insight and helps guide future answer generation.
3. Ensure each principle is general enough to be reused across similar tasks, yet clearly grounded in the specific difference observed.
4. Respond strictly in the following JSON format.

**Input:**
Question: $question
Difference: $diff_list

**Output (JSON only, no extra text):**
{{
  "output": [
    {{
      "Principle": "State a clear and generalizable insight derived from the difference."
    }}
  ]
}}
''')
PRINCIPLE_MATCH_PROMPT = '''
Your goal is to compare the new_principle against each of the existing_principles, and decide one of the following for each:
1. Redundant: if the new and old principle express essentially the same idea.Prefer the newer one.
2. Conflicting: if the two principles provide contradictory guidance. Keep the one that is more general or correct.
3. Irrelevant: if the existing principle is not applicable to the current task anymore. Suggest deletion.
Please return your evaluation in the following JSON format exactly:
{{
    "comparisons": [
        {{
            "old": "<text of the existing principle>",
            "new": "<text of the new principle>",
            "relation": "Redundant | Conflicting | Irrelevant"
        }}
    ]
}}


<input>
New_principle: {new}
Existing_principle:{old}
</input>
'''
