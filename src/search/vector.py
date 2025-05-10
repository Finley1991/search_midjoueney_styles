import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pickle

# 动态生成文件路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, 'data')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置HF模型保存路径变量
os.environ["HF_HOME"] = os.path.join(base_dir, ".cache/huggingface")
text_encoder = SentenceTransformer("richinfoai/ritrieve_zh_v1")  # infgrad/stella-mrl-large-zh-v3.5-1792d

# 加载所有数据
jsonl_file_path = os.path.join(data_dir, 'midjoury_styles_lib_final_zh_en_demo.jsonl')
with open(jsonl_file_path, 'r', encoding="utf-8") as f:
    lines = [json.loads(data.strip()) for data in f.readlines()]

need_embed_keys = [key for key in lines[0].keys() if "zh" in key]

slugs = [line["slug_new"] for line in lines]
img_urls = [line["img_url"] for line in lines]
all_ai_info = [f"""描述: {line['ai_desc_zh']} 风格: {line['ai_style_zh']} 特征: {line['ai_features_zh']}  颜色: {line['ai_color_zh']}""" for line in lines]

vectors_dict = {key: [] for key in need_embed_keys + ["all_ai_info_zh"]}
for line in tqdm(lines):
    slug = line["slug"]
    for key in need_embed_keys:
        texts = [str(line[key])]
        print(key, texts)
        try:
            vectors = text_encoder.encode(texts, normalize_embeddings=False)
            vectors_dict[key].append(normalize(vectors[:, :100])[0])  # d=1792d
        except Exception as e:
            print(f"Error processing slug {slug} for key {key}: {e}")
            vectors_dict[key].append([0.0] * 100)
    key_new = "all_ai_info_zh"
    texts_new = [f"""描述: {line['ai_desc_zh']} 风格: {line['ai_style_zh']} 特征: {line['ai_features_zh']}  颜色: {line['ai_color_zh']}"""]
    print(key_new, texts_new)
    try:
        vectors = text_encoder.encode(texts_new, normalize_embeddings=False)
        vectors_dict[key_new].append(normalize(vectors[:, :100])[0])  # d=1792d
    except Exception as e:
        print(f"Error processing slug {slug} for key {key}: {e}")
        vectors_dict[key].append([0.0] * 100)

# vectors_dict 本地化保存 pkl
vectors_dict_path = os.path.join(data_dir, 'vectors_dict.pkl')
slugs_path = os.path.join(data_dir, 'slugs.pkl')
img_urls_path = os.path.join(data_dir, 'img_urls.pkl')
all_ai_info_path = os.path.join(data_dir, 'all_ai_info.pkl')

with open(vectors_dict_path, "wb") as f:
    pickle.dump(vectors_dict, f)

with open(slugs_path, "wb") as f:
    pickle.dump(slugs, f)

with open(img_urls_path, "wb") as f:
    pickle.dump(img_urls, f)

with open(all_ai_info_path, "wb") as f:
    pickle.dump(all_ai_info, f)