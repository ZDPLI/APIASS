import os
import hashlib
import json
import base64
from dotenv import load_dotenv
load_dotenv()
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

# --- User management ---
USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f)
    except Exception:
        pass


users_db = load_users()


def register_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        return "Имя и пароль обязательны"
    if username in users_db:
        return "Пользователь уже существует"
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    users_db[username] = {"password": pw_hash, "subscription": True}
    save_users(users_db)
    return "Регистрация успешна. Теперь войдите"


def authenticate(username: str, password: str):
    user = users_db.get(username)
    if not user:
        return False, "Неверный логин или пароль"
    if user["password"] != hashlib.sha256(password.encode()).hexdigest():
        return False, "Неверный логин или пароль"
    if not user.get("subscription"):
        return False, "Подписка не активна"
    return True, ""

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

def chat_with_lmstudio(text, image_path=None, history=None, context="", stream=False):
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
        **({"stream": True} if stream else {})
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    if stream:
        resp = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, stream=True, timeout=90)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.strip().startswith("data:"):
                data_str = line.split("data:", 1)[1].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                token = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if token:
                    yield token
        return

    resp = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return reply

def respond(message, image, dialogs, current_dialog):
    history = dialogs.get(current_dialog, [])
    context = retrieve_context(message)
    tokens = []
    for token in chat_with_lmstudio(message, image, history, context, stream=True):
        tokens.append(token)
        yield history + [(message, "".join(tokens))], "", None, dialogs
    reply = "".join(tokens)
    history.append((message, reply))
    dialogs[current_dialog] = history
    yield history, "", None, dialogs


def start_new_dialog(dialogs):
    name = f"Диалог {len(dialogs) + 1}"
    dialogs[name] = []
    return gr.update(choices=list(dialogs.keys()), value=name), dialogs, []


def select_dialog(name, dialogs):
    return dialogs.get(name, [])


def show_landing():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def show_login():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def show_register():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def show_chat():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def do_register(username, password):
    msg = register_user(username, password)
    return msg


def do_login(username, password, user_state, dialogs_state):
    ok, msg = authenticate(username, password)
    if ok:
        dialogs_state = {"Диалог 1": []}
        return (
            "",
            *show_chat(),
            username,
            dialogs_state,
            [],
            gr.update(choices=["Диалог 1"], value="Диалог 1"),
        )
    else:
        return (
            msg,
            *show_login(),
            user_state,
            dialogs_state,
            [],
            gr.update(),
        )


def do_logout():
    return (
        "",
        *show_landing(),
        None,
        {"Диалог 1": []},
        [],
        gr.update(choices=["Диалог 1"], value="Диалог 1"),
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    user_state = gr.State(None)
    dialogs_state = gr.State({"Диалог 1": []})

    landing_box = gr.Column(visible=True)
    with landing_box:
        gr.Markdown("# Добро пожаловать")
        with gr.Row():
            login_nav = gr.Button("Войти")
            register_nav = gr.Button("Регистрация")

    login_box = gr.Column(visible=False)
    with login_box:
        gr.Markdown("## Вход")
        login_msg = gr.Markdown("")
        login_user = gr.Textbox(label="Имя")
        login_pass = gr.Textbox(type="password", label="Пароль")
        with gr.Row():
            login_submit = gr.Button("Войти")
            login_back = gr.Button("Назад")

    register_box = gr.Column(visible=False)
    with register_box:
        gr.Markdown("## Регистрация")
        register_msg = gr.Markdown("")
        register_user = gr.Textbox(label="Имя")
        register_pass = gr.Textbox(type="password", label="Пароль")
        with gr.Row():
            register_submit = gr.Button("Создать аккаунт")
            register_back = gr.Button("Назад")

    chat_box = gr.Column(visible=False)
    with chat_box:
        gr.Markdown("# Мультимодальный медицинский ассистент")
        logout_btn = gr.Button("Выйти")
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

    login_nav.click(lambda: show_login(), None, [landing_box, login_box, register_box, chat_box])
    register_nav.click(lambda: show_register(), None, [landing_box, login_box, register_box, chat_box])
    login_back.click(lambda: show_landing(), None, [landing_box, login_box, register_box, chat_box])
    register_back.click(lambda: show_landing(), None, [landing_box, login_box, register_box, chat_box])

    register_submit.click(do_register, [register_user, register_pass], register_msg)
    login_submit.click(
        do_login,
        [login_user, login_pass, user_state, dialogs_state],
        [
            login_msg,
            landing_box,
            login_box,
            register_box,
            chat_box,
            user_state,
            dialogs_state,
            chatbot,
            dialog_select,
        ],
    )
    logout_btn.click(
        do_logout,
        None,
        [
            login_msg,
            landing_box,
            login_box,
            register_box,
            chat_box,
            user_state,
            dialogs_state,
            chatbot,
            dialog_select,
        ],
    )

    file.upload(load_document, file, None)
    new_btn.click(start_new_dialog, dialogs_state, [dialog_select, dialogs_state, chatbot])
    dialog_select.change(select_dialog, [dialog_select, dialogs_state], chatbot)
    send.click(respond, [txt, img, dialogs_state, dialog_select], [chatbot, txt, img, dialogs_state])

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
