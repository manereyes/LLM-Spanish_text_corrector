import streamlit as st
import time
from llm import invoke_ai

### HEADER ###
st.title("Corrector de texto")
st.text("Para pruebas")

### SIDE BAR ###
with st.sidebar:
    st.subheader("Model")
    model = st.selectbox(
        "select a model.",
        ("llama3.1", "mistral")
    )

    st.subheader("Temperature")
    temp = st.slider(
        "Increases creativity at higher values",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=0.0
    )

    st.subheader("Mirostat")
    mirostat = st.slider(
        "Reduces perplexity",
        min_value=0,
        max_value=2,
        step=1,
        value=1
    )

    st.subheader("Mirostat ETA")
    mirostat_eta = st.slider(
        "Respond to feedback of generated text",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        value=0.1
    )

    st.subheader("Mirostat TAU")
    mirostat_tau = st.slider(
        "Controls balance between coherence and diversity",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=5.0
    )

    st.subheader("Num CTX")
    num_ctx = int(st.text_input(
        "Size of context window to generate next token",
        "2048"
    ))

    st.subheader("Top k")
    top_k = int(st.slider(
        "Reduces probability of generating nonsense",
        min_value=1,
        max_value=150,
        step=1,
        value=40
    ))

    st.subheader("Top p")
    top_p = st.slider(
        "Controls between focused text and diverse text",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=0.9
    )

### BODY ###
with st.container():
    input = st.text_area(
        "Texto a corregir:"
    )
    if st.button("Test Response"):
        with st.spinner("Generating response..."):
            response = invoke_ai(model, input, temp, mirostat, mirostat_eta, mirostat_tau, num_ctx, top_k, top_p)
            st.header("Correcciones gramaticales.")
            st.write(response)