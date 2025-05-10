import gradio as gr
import requests
import base64

# Define API endpoints
BASE_URL = "http://127.0.0.1:8000"
PIC_CAPTION_URL = f"{BASE_URL}/api/v1/pic_caption/url"
PIC_CAPTION_FILE = f"{BASE_URL}/api/v1/pic_caption/file"
EMBEDDING_URL = f"{BASE_URL}/api/v1/embedding"
STYLE_SEARCH_URL = f"{BASE_URL}/api/v1/style_search"

# Function to handle the inputs and call the appropriate API
def process_input(text, img_url, img_file):
    uploaded_image_html = ""
    if img_file:
        # Display uploaded image
        with open(img_file.name, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        uploaded_image_html = f'''
        <div style="text-align: center; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: #f9f9f9;">
            <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: auto; border-radius: 8px;">
        </div>
        '''
    elif img_url:
        # Display image from URL
        uploaded_image_html = f'''
        <div style="text-align: center; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: #f9f9f9;">
            <img src="{img_url}" style="width: 100%; height: auto; border-radius: 8px;">
        </div>
        '''

    if text:
        embedding_response = requests.post(EMBEDDING_URL, json={"text": text})
        embedding_data = embedding_response.json()
        query_vector = embedding_data["vector"]

        style_search_response = requests.post(STYLE_SEARCH_URL, json={
            "query_vector": query_vector,
            "search_type": "all_ai_info",
            "k": 50
        })
        style_search_data = style_search_response.json()
        return uploaded_image_html, format_results(style_search_data)

    elif img_url:
        response = requests.post(PIC_CAPTION_URL, json={"img_url": img_url})
        caption_data = response.json()

        combined_text = f"{caption_data['desc']} {caption_data['style']} {caption_data['features']} {caption_data['color']}"

        embedding_response = requests.post(EMBEDDING_URL, json={"text": combined_text})
        embedding_data = embedding_response.json()
        query_vector = embedding_data["vector"]

        style_search_response = requests.post(STYLE_SEARCH_URL, json={
            "query_vector": query_vector,
            "search_type": "all_ai_info",
            "k": 6
        })
        style_search_data = style_search_response.json()
        return uploaded_image_html, format_results(style_search_data)

    elif img_file:
        with open(img_file.name, "rb") as f:
            files = {"file": f}
            response = requests.post(PIC_CAPTION_FILE, files=files)
        caption_data = response.json()

        combined_text = f"{caption_data['desc']} {caption_data['style']} {caption_data['features']} {caption_data['color']}"

        embedding_response = requests.post(EMBEDDING_URL, json={"text": combined_text})
        embedding_data = embedding_response.json()
        query_vector = embedding_data["vector"]

        style_search_response = requests.post(STYLE_SEARCH_URL, json={
            "query_vector": query_vector,
            "search_type": "all_ai_info",
            "k": 6
        })
        style_search_data = style_search_response.json()
        return uploaded_image_html, format_results(style_search_data)

    else:
        return "Please provide text, an image URL, or upload an image.", ""

def format_results(style_search_data):
    results = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">'
    for img_url, desc in zip(style_search_data["img_urls"], style_search_data["desc"]):
        results += f'''
        <div style="text-align: center; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: #f9f9f9;">
            <img src="{img_url}" alt="{desc}" style="width: 100%; height: auto; border-radius: 8px;">
            <p style="margin-top: 10px; font-size: 14px; color: #333;">{desc}</p>
        </div>
        '''
    results += '</div>'
    return results

# Gradio interface
with gr.Blocks(css=".gradio-container { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; }") as demo:
    gr.Markdown("""
    <h1 style="text-align: center; color: #4CAF50;">Image Caption and Style Search</h1>
    <p style="text-align: center; font-size: 16px; color: #555;">Upload an image, provide an image URL, or enter a text description to explore styles and captions.</p>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Text Description", placeholder="Enter text for embedding and style search", lines=2)
            img_url_input = gr.Textbox(label="Image URL", placeholder="Enter image URL for captioning")
            img_file_input = gr.File(label="Upload Image", file_types=["image"])
            preview_image = gr.HTML(label="Preview Image")
            submit_button = gr.Button("Submit", elem_id="submit-button")
        with gr.Column(scale=2):
            output = gr.HTML(label="Results")

    submit_button.click(
        process_input,
        inputs=[text_input, img_url_input, img_file_input],
        outputs=[preview_image, output]
    )

# Launch the Gradio app
demo.launch(server_port=8001)