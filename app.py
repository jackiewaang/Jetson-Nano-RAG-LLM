import streamlit as st
import requests
import psutil

st.title("Local RAG Chatbot Assistant")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Advanced settings for model parameters and modes
with st.expander("Advanced Settings", expanded=False):
    mode = st.radio("Choose mode:", ("RAG", "Chat"))
    k = st.slider(
            "K - Number of document segments retrieved",
            min_value=1, max_value=20, value=10, disabled=(mode == "Chat"),
            help="How many text chunks the retriever should fetch from the knowledge base before ranking them. Larger K = wider search, but slower and possibly less relevant results.")
    n = st.slider(
            "N - Top document segments used",
            min_value=1, max_value=k, value=3, disabled=(mode == "Chat"),
            help="From the K retrieved chunks, the top N most relevant ones are selected and sent to the LLM as context. Smaller N = more focused context, larger N = more coverage.")
    max_tokens = st.number_input(
            "Max tokens - Maximum response length",
            min_value=-1, max_value=1000, value=-1, step=50,
            help="Sets upper limit on how many tokens the model can generate in a single reply. Large values = longer answers, but more computation and memory use. Small value = shorter, faster, but possibly truncated answers. -1 = no limit")
    temperature = st.slider(
            "Temperature - Randomness/Creativity",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1,
            help="Parameter that controls the randomness and creativity of model's output. 0 = more predictable and deterministic (for factual, technical tasks), 1 = more creative and unexpected results (for open-ended tasks like storytelling/brainstorming")

st.write(f"Using k={k}, n={n}, max_tokens={max_tokens}, temperature={temperature}")

# Upload multiple PDF files
uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
)

# Send files to backend for processing
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

# Chat form for user queries
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Query", key="input", placeholder= "Ask anything here...")
    submitted = st.form_submit_button("Send")
    
    
    if submitted and prompt:
        # Choose endpoint based on mode
        endpoint = "http://localhost:8000/"
        if mode == "Chat":
            endpoint = "http://localhost:8000/chat"
            # Chat endpoint
            response = requests.post(
                endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
        else:
            # RAG endpoint
            response = requests.post(
                endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "k": k,
                    "n": n
                }
            )
        
        data = response.json()
        
        # Extract answer and sources
        choices = data.get("choices", [])
        answer = ""
        if choices:
            message = choices[0].get("message", {})
            answer = message.get("content", "")
        
        sources_used = data.get("sources_used", {})
        source_files = sources_used.get("files", [])
        detailed_sources = sources_used.get("detailed_sources", [])

        # Extract performance metrics
        timings = data.get("timings", {})
        inference_time = (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000
    
        usage = data.get("usage", {})
        ctx_len = usage.get("prompt_tokens", 0)

        throughput = timings.get("predicted_per_second", 0)

        memory_usage = 0
        for proc in psutil.process_iter(['name', 'memory_info']):
            if "llama" in proc.info['name'].lower():
                memory_usage += proc.info['memory_info'].rss

        # Append metrics to session state messages
        st.session_state.messages.append({
            "user": prompt,
            "LLM": answer,
            "sources_referenced": source_files,
            "detailed_sources": detailed_sources,
            "time": inference_time,
            "throughput": throughput,
            "ctx_len": ctx_len,
            "memory": memory_usage
            })

for chat in reversed(st.session_state.messages):
    st.markdown(f"User: {chat['user']}")
    st.markdown(f"AI: {chat['LLM']}")
    # Display sources if available
    if chat.get('sources_referenced') and len(chat['sources_referenced']) > 0:
        st.markdown("**Sources Used:**")
        for source in chat['sources_referenced']:
            # Extract just the filename 
            filename = source.split('/')[-1] if '/' in source else source
            st.markdown(f"• {filename}")
        
        # Display each source with its page number
        if chat.get('detailed_sources'):
            with st.expander("Detailed Source Information"):
                for source in chat['detailed_sources']:
                    filename = source['source'].split('/')[-1] if '/' in source['source'] else source['source']
                    page = source.get('page', 'N/A')
                    st.markdown(f"• **{filename}**: Page {page}")
    st.markdown(f"Inference time: {chat['time']:.2f} seconds")
    st.markdown(f"Throughput: {chat['throughput']:.2f} tokens/second")
    st.markdown(f"Context Length: {chat['ctx_len']} tokens")
    st.markdown(f"Memory Usage: {chat['memory'] / (1024**3):.2f}GB")
    st.markdown("---")
