import requests
import base64
from PIL import Image
from io import BytesIO

# 定义全局变量
OLLAMA_API = "http://127.0.0.1:11434/api/chat" # Ollama API 地址, 需要调用多模态大模型
PROMPT_CAPTION = """
请简单描述一下这个图片，包括以下几个部分：
[画面内容]: 图片的具体内容描述，
[图片风格]: 图片风格，
[图片特征]: 图片特征，
[图片色彩]: 图片色彩
返回格式要求json：
key为desc，value为图片内容，string，
key为style，value为图片风格，string，
key为features，value为图片特征，string，
key为color，value为图片色彩，string。
1. 严格按照格式返回
2. 不要返回其他任何内容
"""

def pic_caption(prompt, local_img_base64):
    """
    调用 Ollama API 进行图片描述
    """
    try:
        data = {
            "model": "gemma3:27b",
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [local_img_base64] if local_img_base64 else [],
                }
            ]
        }

        resp = requests.post(OLLAMA_API, json=data)
        resp.raise_for_status()  # 检查 HTTP 请求是否成功
        result = []

        for item in resp.text.split("\n"):
            try:
                parsed_item = eval(item.replace("false", "False").replace("true", "True"))
                if "message" in parsed_item and "content" in parsed_item["message"]:
                    result.append(parsed_item["message"]["content"])
            except (SyntaxError, NameError):
                continue

        return "".join(result).replace("`", "").replace("json", "")
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""


def load_image_from_url(img_url):
    """
    从 URL 加载图片
    """
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Failed to load image from URL: {e}")
        return None


def image_to_base64(img_url):
    """
    将图片 URL 转换为 Base64 编码
    """
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        img = BytesIO(response.content)
        return base64.b64encode(img.getvalue()).decode('utf-8')
    except requests.RequestException as e:
        print(f"Failed to convert image to Base64: {e}")
        return None


if __name__ == "__main__":
    # 示例图片 URL
    img_url = "https://img2.baidu.com/it/u=2781037149,3187912571&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=1422"

    # 加载图片并转换为 Base64
    img_base64 = image_to_base64(img_url)
    if img_base64:
        # 调用图片描述函数
        result = pic_caption(PROMPT_CAPTION, img_base64)
        print(result)