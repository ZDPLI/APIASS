import os
from dotenv import load_dotenv
load_dotenv()
import base64
import requests
import gradio as gr

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://172.23.32.1:1234")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.environ.get("LMSTUDIO_MODEL", "lingshu-7b")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")

if not LMSTUDIO_URL:
    raise RuntimeError("LMSTUDIO_URL environment variable not set")

CHAT_ENDPOINT = LMSTUDIO_URL.rstrip('/') + "/v1/chat/completions"

def chat_with_lmstudio(text, image_path=None, history=None):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    user_content = []
    if text:
        user_content.append({"type": "text", "text": text})
    if image_path:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages.append({"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]})

    payload = {
        "model": MODEL,
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return reply

def respond(message, image, history):
    reply = chat_with_lmstudio(message, image, history)
    history.append((message, reply))
    return history, "", None

with gr.Blocks() as demo:
    gr.Markdown("# Мультимодальный медицинский ассистент")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(label="Сообщение")
        img = gr.Image(type="filepath", label="Изображение (необязательно)")
    send = gr.Button("Отправить")
    send.click(respond, [txt, img, chatbot], [chatbot, txt, img])

    demo.launch(server_name="0.0.0.0", server_port=7860)
