import streamlit as st
import requests
import psutil

st.title("Local RAG Chatbot Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_ollama_models():
    response = requests.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in response.json().get("models", [])]
    return models

models = get_ollama_models()

if models:
    selected_model = st.selectbox("Choose a model", models)
    st.write(f"You selected: {selected_model}")
else:
    st.warning("No models found locally")

uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
)

if uploaded_files and st.button("Upload to backend"):
    files = [
            ("files", (file.name, file.getvalue(), "application/pdf"))
            for file in uploaded_files
    ]

    with st.spinner("Uploading and processing..."):
        response = requests.post("http://localhost:8000/upload", files=files)

    if response.status_code == 200:
        st.success(response.json().get("message", "Upload Successful!"))
    else:
        st.error(f"Upload failed: {response.text}")


with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Query", key="input", placeholder= "Ask anything here...")
    submitted = st.form_submit_button("Send")
    
    if submitted and prompt:
        response = requests.post(
            "http://localhost:8000/",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
            }
        )
        inference_time = response.json().get("total_duration", 0) / 1e9
        answer = response.json().get("response", "")
        throughput = response.json().get("eval_count") / (response.json().get("eval_duration") / 1e9)
        ctx_len = len(response.json().get("context"))

        memory_usage = 0
        for proc in psutil.process_iter(['name', 'memory_info']):
            if "ollama" in proc.info['name'].lower():
                memory_usage += proc.info['memory_info'].rss

        st.session_state.messages.append({
            "user": prompt,
            "LLM": answer,
            "time": inference_time,
            "throughput": throughput,
            "ctx_len": ctx_len,
            "memory": memory_usage
            })

for chat in reversed(st.session_state.messages):
    st.markdown(f"User: {chat['user']}")
    st.markdown(f"AI: {chat['LLM']}")
    st.markdown(f"Inference time: {chat['time']:.2f} seconds")
    st.markdown(f"Throughput: {chat['throughput']:.2f} tokens/second")
    st.markdown(f"Context Length: {chat['ctx_len']} tokens")
    st.markdown(f"Memory Usage: {chat['memory'] / (1024**3):.2f}GB")
    st.markdown("---")

