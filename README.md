# Multimodal Medical Assistant

This project implements a small Gradio based chat system that works with LMStudio API. It allows account creation and authentication, supports multiple dialogues, and can use uploaded text documents as additional context for the assistant.

## Features

- **LMStudio integration**. Requests are sent to the LMStudio API endpoint defined by `LMSTUDIO_URL`.
- **User management**. Users are stored in a JSON file (`users.json`). Subscription status is checked at login.
- **Authentication pages**. Landing, login, registration and chat views are implemented in Gradio with simple navigation logic.
- **Streaming responses**. Chat replies are streamed token by token so the interface updates while the model is generating output.
- **Dialogue history management**. Users can maintain several dialogues and switch between them via a dropdown list.
- **RAG (Retrieval Augmented Generation)**. `.txt` and `.docx` files can be uploaded. Their contents are embedded with `SentenceTransformer` and retrieved based on user queries.

## Running

1. Install the required packages (for example via pip):
   ```bash
   pip install gradio python-dotenv sentence-transformers python-docx
   ```
   If the environment restricts network access, install these packages via an alternate method or pre-built wheels.

2. Set the following environment variables as needed:
   - `LMSTUDIO_URL` – base URL for LMStudio API.
   - `LMSTUDIO_API_KEY` – token for API access.
   - `LMSTUDIO_MODEL` – name of the model.
   - `SYSTEM_PROMPT` – optional system prompt sent with every conversation.

3. Launch the app:
   ```bash
   python app.py
   ```
   The interface will be available on port `7860` by default.

## Usage

- Visit the landing page and register a new account, or log in if you already have one.
- After login you can start chatting immediately. Use the *Новый диалог* button to create additional dialogues.
- Upload `.txt` or `.docx` files to make their contents available for retrieval. The assistant will use the most similar text snippets as additional context when generating replies.

## Data storage

User information is stored in `users.json`. Each entry contains a SHA-256 password hash and a simple boolean subscription flag. Document embeddings are kept in memory only and lost when the server restarts.

