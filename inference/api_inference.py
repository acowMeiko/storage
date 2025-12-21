import requests
import config
def gpt_call(user, *, 
             url: str = None,
             model: str = None,
             system_message: str = "You are a helpful assistant.",
             api_key: str = None,
             temperature: float = None,
             top_p: float = None) -> str:
    """调用GPT模型的API接口（支持 OpenAI 兼容格式）"""
    # 使用默认配置
    base_url = url or config.BASE_MODEL_API_URL
    model = model or config.BASE_MODEL_NAME
    temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
    top_p = top_p if top_p is not None else config.DEFAULT_TOP_P
    
    # 构建完整端点：如果 base_url 以 /v1/ 结尾，添加 chat/completions
    if base_url.endswith('/v1/'):
        endpoint = base_url + 'chat/completions'
    elif base_url.endswith('/v1'):
        endpoint = base_url + '/chat/completions'
    else:
        endpoint = base_url  # 假设已经是完整端点
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif hasattr(gpt_call, 'default_api_key'):
        headers["Authorization"] = f"Bearer {gpt_call.default_api_key}"

    # 构建请求载荷
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # 提取响应内容
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            return message.get("content", "").strip()
        else:
            print(f"[WARNING] Unexpected API response format: {result}")
            return ""

    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        if 'response' in locals():
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response content: {response.text}")
        return ""
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return ""