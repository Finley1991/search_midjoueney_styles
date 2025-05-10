# Midjourney 图片搜索库

该项目允许用户通过图片或文本描述搜索相似的图片。
项目效果图如下：
![](./data/img/pic1.png)

![](./data/img/pic2.png)
以下是项目的详细信息和使用说明。

## 环境要求
确保你已经安装了以下依赖，可以通过 `requirements.txt` 文件安装：
```bash
pip install -r requirements.txt
```
## 数据要求
参考数据依赖准备部分，确保你已经准备好数据集。

### 主要依赖
- FastAPI: 用于构建 API 服务
- Gradio: 用于构建 Web 界面
- SentenceTransformer: 用于文本嵌入
- FAISS: 用于向量搜索
- SQLite3: 用于数据存储

## 运行命令
1. 启动 FastAPI 服务：
   ```bash
   uvicorn src.api.service:app --reload
   ```

2. 启动 Gradio 应用：
   ```bash
   python src/webui/webui.py
   ```

## 测试案例
### 通过图片 URL 搜索
在 Gradio 界面的 Image URL 输入框中输入图片的 URL，点击 Submit 按钮，系统将生成图片描述并搜索相似图片。

### 通过文本描述搜索
在 Gradio 界面的 Text Description 输入框中输入文本描述，点击 Submit 按钮，系统将生成文本嵌入并搜索相似图片。

### 上传图片搜索
点击 Upload Image 按钮上传本地图片，点击 Submit 按钮，系统将生成图片描述并搜索相似图片。

## 项目结构
   
   ```
   midjourney_library/
   ├── data/ (包含所有数据文件)
   │   ├── all_ai_info.pkl  # 所有图片的 AI 信息
   │   ├── img_urls.pkl     # 所有图片的 URL
   │   ├── index4all_ai_info.faiss  # 所有图片的 AI 信息索引
   │   ├── index4color.faiss  # 所有图片的颜色索引
   │   ├── index4content.faiss  # 所有图片的内容索引
   │   ├── index4features.faiss  # 所有图片的特征索引
   │   ├── index4style.faiss  # 所有图片的风格索引
   │   ├── index4type.faiss  # 所有图片的类型索引
   │   ├── midjourney_styles.db  # 数据库文件
   │   ├── midjoury_styles_lib_final_zh_en.jsonl  # 源文件，只公开500条数据，包含图片url，AI描述（Gemma3-27b多模态推理）等
   │   ├── slugs.pkl  # 所有图片的 slug
   │   └── vectors_dict.pkl  # 所有图片的向量
   ├── src/ (包含所有源代码)
   │   ├── api/ (包含 API 相关的代码)
   │   │   └── service.py  # FastAPI 服务代码
   │   ├── database/ (包含数据库相关的代码)
   │   │   └── process_data.py  # 数据库处理代码
   │   ├── image_processing/ (包含图片处理相关的代码)
   │   │   └── ollama_picture_desc.py  # 图片描述代码
   │   ├── search/ (包含搜索相关的代码)
   │   │   ├── style_search.py  # 风格搜索代码
   │   │   └── vector.py  # 向量搜索代码
   │   └── webui/ (包含 Web 界面相关的代码)
   │       └── webui.py  # Gradio 应用代码
   ├── requirements.txt  # 项目依赖文件
   └── README.md  # 项目说明文件
   ```

## 数据依赖准备
1. 下载数据集
   - 下载 midjourney_styles_lib_final_zh_en_demo.json，并将其放置在 `data/` 目录下。

2. 运行以下命令以处理数据并生成数据库：
   ```bash
   python src/database/process_data.py
   ```
    - 该脚本将读取 `data/midjourney_styles_lib_final_zh_en_demo.json` 文件，处理数据并生成sqlit3数据库。
    - 处理完成后，数据库文件将保存在 `data/` 目录下。
3. 运行以下命令生成向量和建立faiss索引
   ```bash
   python src/search/vector.py
   ```
    - 该脚本将读取`data/midjourney_styles_lib_final_zh_en_demo.json` 文件，生成向量。
    - 处理完成后，向量文件将保存在 `data/` 目录下。
4. 运行一下命令建立索引
   ```bash
   python src/search/style_search.py
   ```
    - 该脚本将读取`data/*.pkl` 文件，建立索引。
    - 处理完成后，索引文件将保存在 `data/` 目录下。


