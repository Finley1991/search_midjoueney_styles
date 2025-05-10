from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from pydantic import BaseModel
from ..image_processing.ollama_picture_desc import image_to_base64, pic_caption, PROMPT_CAPTION
from ..search.style_search import all_ai_info, slugs, img_urls, search_index, index4content, index4style, \
    index4features, index4color, index4all_ai_info
from sentence_transformers import SentenceTransformer
import numpy as np
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import json
from typing import Optional
import validators  # For URL validation
from sklearn.preprocessing import normalize

app = FastAPI()

# Initialize text embedding model
text_encoder = SentenceTransformer("richinfoai/ritrieve_zh_v1")


# Request and response models
class PicCaptionRequest(BaseModel):
    img_url: Optional[str] = None


class PicCaptionResponse(BaseModel):
    desc: str
    style: str
    features: str
    color: str


class StyleSearchRequest(BaseModel):
    query_vector: list[float]
    search_type: str  # e.g., "content", "style", "features", "color", "all_ai_info"
    k: int = 5  # Number of results to return


class StyleSearchResponse(BaseModel):
    distances: list[float]
    indices: list[int]
    slugs: list[str]
    img_urls: list[str]
    desc: list[str]


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    vector: list[float]


# 处理 URL 请求
@app.post("/api/v1/pic_caption/url", response_model=PicCaptionResponse)
async def generate_pic_caption_by_url(request: PicCaptionRequest):
    if not request.img_url:
        raise HTTPException(status_code=400, detail="img_url is required")

    # 验证并处理图像 URL
    if not validators.url(request.img_url):
        raise HTTPException(status_code=400, detail="Invalid image URL.")
    img_base64 = image_to_base64(request.img_url)
    if not img_base64:
        raise HTTPException(status_code=400, detail="Failed to process the image URL.")

    # 生成描述
    result = pic_caption(PROMPT_CAPTION, img_base64)
    try:
        result_json = json.loads(result)
        return PicCaptionResponse(
            desc=result_json.get("desc", ""),
            style=result_json.get("style", ""),
            features=result_json.get("features", ""),
            color=result_json.get("color", "")
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse the response: {e}")


# 处理文件上传请求
@app.post("/api/v1/pic_caption/file", response_model=PicCaptionResponse)
async def generate_pic_caption_by_file(file: UploadFile = File(...)):
    # 处理上传的图像文件
    try:
        image = Image.open(BytesIO(await file.read()))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process the uploaded image: {e}")

    # 生成描述
    result = pic_caption(PROMPT_CAPTION, img_base64)
    try:
        result_json = json.loads(result)
        return PicCaptionResponse(
            desc=result_json.get("desc", ""),
            style=result_json.get("style", ""),
            features=result_json.get("features", ""),
            color=result_json.get("color", "")
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse the response: {e}")


# API to perform style search
@app.post("/api/v1/style_search", response_model=StyleSearchResponse)
def style_search(request: StyleSearchRequest):
    # Map search type to the corresponding FAISS index
    index_map = {
        "content": index4content,
        "style": index4style,
        "features": index4features,
        "color": index4color,
        "all_ai_info": index4all_ai_info
    }
    index = index_map.get(request.search_type)
    if not index:
        raise HTTPException(status_code=400, detail="Invalid search type provided.")

    query_vector = np.array(request.query_vector, dtype="float32").reshape(1, -1)
    if query_vector.shape[1] != index.d:
        raise HTTPException(status_code=400, detail="Query vector dimension does not match index dimension.")

    distances, indices = search_index(index, query_vector, request.k)
    r_idx = indices[0].tolist()
    r_distances = distances[0].tolist()
    r_slugs = [slugs[i] for i in r_idx]
    r_img_urls = [img_urls[i] for i in r_idx]
    r_all_ai_info = [all_ai_info[i] for i in r_idx]

    # 这里r_img_urls会有重复的url，需要去重，
    new_r_img_urls = []
    new_r_slugs = []
    new_r_all_ai_info = []
    new_r_distances = []
    new_r_idx = []
    for distance_, slug_, img_url_, all_ai_info_, r_idx_ in zip(r_distances, r_slugs, r_img_urls, r_all_ai_info, r_idx):
        if img_url_ not in new_r_img_urls:
            new_r_img_urls.append(img_url_)
            new_r_slugs.append(slug_)
            new_r_all_ai_info.append(all_ai_info_)
            new_r_distances.append(distance_)
            new_r_idx.append(r_idx_)

    return StyleSearchResponse(
        distances=new_r_distances[:6],
        indices=new_r_idx[:6],
        slugs=new_r_slugs[:6],
        img_urls=new_r_img_urls[:6],
        desc=new_r_all_ai_info[:6],
    )


# API to generate text embedding
@app.post("/api/v1/embedding", response_model=EmbeddingResponse)
def generate_embedding(request: EmbeddingRequest):
    try:
        vector = text_encoder.encode([request.text], normalize_embeddings=False)
        cut_vector = normalize(vector[:, :100])[0]  # Reduce to 100 dimensions
        return EmbeddingResponse(vector=cut_vector.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")
