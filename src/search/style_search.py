import faiss
import pickle
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 动态生成文件路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, 'data')

# 检查并创建 data 目录
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load pickle files
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

vector_dict_path = os.path.join(data_dir, 'vectors_dict.pkl')
slugs_path = os.path.join(data_dir, 'slugs.pkl')
img_urls_path = os.path.join(data_dir, 'img_urls.pkl')
all_ai_info_path = os.path.join(data_dir, 'all_ai_info.pkl')

vector_dict = load_pkl(vector_dict_path)
slugs = load_pkl(slugs_path)
img_urls = load_pkl(img_urls_path)
all_ai_info = load_pkl(all_ai_info_path)

# Build FAISS index
def build_index(vectors, index_type="flat", save_path=None):
    """
    Build a FAISS index for the given vectors.

    Args:
        vectors (list): List of vectors to index.
        index_type (str): Type of FAISS index to use ("flat" or "ivf").
        save_path (str): Path to save the built index (optional).

    Returns:
        faiss.Index: The built FAISS index.
    """
    if not vectors or len(vectors) == 0:
        raise ValueError("The input vectors are empty or invalid.")

    d = len(vectors[0])  # Vector dimension
    vectors_np = np.array(vectors).astype('float32')

    if index_type == "flat":
        index = faiss.IndexFlatL2(d)  # Flat L2 index
    elif index_type == "ivf":
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(vectors_np)  # Train the IVF index
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(vectors_np)  # Add vectors to the index
    logging.info(f"Built {index_type} index with {len(vectors)} vectors.")

    if save_path:
        faiss.write_index(index, save_path)
        logging.info(f"Index saved to {save_path}.")

    return index

# Load or build index
def load_or_build_index(vectors, index_path, index_type="flat"):
    """
    Load a FAISS index from a file or build it if not available.

    Args:
        vectors (list): List of vectors to index.
        index_path (str): Path to the index file.
        index_type (str): Type of FAISS index to use ("flat" or "ivf").

    Returns:
        faiss.Index: The loaded or built FAISS index.
    """
    if os.path.exists(index_path):
        logging.info(f"Loading index from {index_path}.")
        return faiss.read_index(index_path)
    else:
        logging.info(f"Index file not found. Building a new {index_type} index.")
        return build_index(vectors, index_type=index_type, save_path=index_path)

# Build or load indices for different vector types
index4type_path = os.path.join(data_dir, 'index4type.faiss')
index4content_path = os.path.join(data_dir, 'index4content.faiss')
index4style_path = os.path.join(data_dir, 'index4style.faiss')
index4features_path = os.path.join(data_dir, 'index4features.faiss')
index4color_path = os.path.join(data_dir, 'index4color.faiss')
index4all_ai_info_path = os.path.join(data_dir, 'index4all_ai_info.faiss')

index4type = load_or_build_index(vector_dict["type_zh"], index4type_path, index_type="flat")
index4content = load_or_build_index(vector_dict["desc_zh"], index4content_path, index_type="flat")
index4style = load_or_build_index(vector_dict["ai_style_zh"], index4style_path, index_type="flat")
index4features = load_or_build_index(vector_dict["ai_features_zh"], index4features_path, index_type="flat")
index4color = load_or_build_index(vector_dict["ai_color_zh"], index4color_path, index_type="flat")
index4all_ai_info = load_or_build_index(vector_dict["all_ai_info_zh"], index4all_ai_info_path, index_type="flat")

# FAISS search function
def search_index(index, query_vector, k=5):
    """
    Search the FAISS index for the k most similar vectors.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_vector (np.ndarray): The query vector.
        k (int): Number of results to return.

    Returns:
        tuple: Distances and indices of the top-k results.
    """
    D, I = index.search(query_vector, k)  # D: distances, I: indices
    return D, I
