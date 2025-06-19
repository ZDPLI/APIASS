import os
from dotenv import load_dotenv
load_dotenv()
import base64
import requests
import gradio as gr
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://172.23.32.1:1234")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.environ.get("LMSTUDIO_MODEL", "lingshu-7b")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")

if not LMSTUDIO_URL:
    raise RuntimeError("LMSTUDIO_URL environment variable not set")

CHAT_ENDPOINT = LMSTUDIO_URL.rstrip('/') + "/v1/chat/completions"

# --- Retrieval system setup ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_chunks = []
doc_embeddings = None


def load_document(path):
    """Load text from docx or txt file and update embeddings."""
    global doc_chunks, doc_embeddings
    if path.endswith(".docx"):
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # simple split
    import textwrap
    chunks = [c.strip() for c in textwrap.wrap(text, 500) if c.strip()]
    if not chunks:
        return

    emb = embedder.encode(chunks)
    if doc_embeddings is None:
        doc_chunks = chunks
        doc_embeddings = emb
    else:
        doc_chunks.extend(chunks)
        doc_embeddings = np.vstack([doc_embeddings, emb])


def retrieve_context(query, top_n=3):
    if doc_embeddings is None:
        return ""
    q_emb = embedder.encode([query])
    sims = np.dot(doc_embeddings, q_emb.T).squeeze()
    idx = np.argsort(-sims)[:top_n]
    selected = [doc_chunks[i] for i in idx if sims[i] > 0]
    return "\n".join(selected)

def chat_with_lmstudio(text, image_path=None, history=None, context=""):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if history:
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

    if context:
        messages.append({"role": "system", "content": f"Additional context:\n{context}"})

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

def respond(message, image, dialogs, current_dialog):
    history = dialogs.get(current_dialog, [])
    context = retrieve_context(message)
    reply = chat_with_lmstudio(message, image, history, context)
    history.append((message, reply))
    dialogs[current_dialog] = history
    return history, "", None, dialogs


def start_new_dialog(dialogs):
    name = f"Диалог {len(dialogs) + 1}"
    dialogs[name] = []
    return gr.Dropdown.update(choices=list(dialogs.keys()), value=name), dialogs, []


def select_dialog(name, dialogs):
    return dialogs.get(name, [])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Мультимодальный медицинский ассистент")

    dialogs_state = gr.State({"Диалог 1": []})

    with gr.Row():
        with gr.Column(scale=1):
            dialog_select = gr.Dropdown(label="Диалог", choices=["Диалог 1"], value="Диалог 1")
            new_btn = gr.Button("Новый диалог")
            file = gr.File(file_types=[".txt", ".docx"], label="Документ")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                txt = gr.Textbox(label="Сообщение")
                img = gr.Image(type="filepath", label="Изображение")
            send = gr.Button("Отправить")

    file.upload(load_document, file, None)
    new_btn.click(start_new_dialog, dialogs_state, [dialog_select, dialogs_state, chatbot])
    dialog_select.change(select_dialog, [dialog_select, dialogs_state], chatbot)
    send.click(respond, [txt, img, dialogs_state, dialog_select], [chatbot, txt, img, dialogs_state])

    demo.launch(server_name="0.0.0.0", server_port=7860)
