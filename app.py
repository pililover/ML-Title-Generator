import streamlit as st
from dotenv import load_dotenv
import os

from underthesea import word_tokenize
from transformers import EncoderDecoderModel, AutoTokenizer
import torch

st.set_page_config(page_title="Trình sinh tiêu đề", layout="centered")

torch.classes.__path__ = [] # add this line to manually set it to empty. 

# Load pre-trained model và tokenizer nhe
@st.cache_resource

def load_model_and_tokenizer():
    model_name = "PuppetLover/Title_generator"  # Hugging Face model repo
    tokenizer_name = "PuppetLover/Finetune_PhoBert"   # Tokenizer for PhoBERT
    token = os.getenv("HUGGINGFACE_TOKEN")  

    # Load lên
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = EncoderDecoderModel.from_pretrained(model_name, use_auth_token=token)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Dưới này là app configuration nè
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

with st.sidebar:
    if st.button("🧾 Hiện/Ẩn lịch sử"):
        st.session_state.show_sidebar = not st.session_state.show_sidebar

if st.session_state.show_sidebar:
    with st.sidebar:
        st.markdown("### 🕓 Lịch sử")
        st.write("- Cuộc trò chuyện 1: *Tiêu đề sinh từ đoạn A*")
        st.write("- Cuộc trò chuyện 2: *Tiêu đề sinh từ đoạn B*")
        st.write("- ...")

st.markdown("""
    <style>
        textarea {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
            border: 1px solid #ffffff30;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #444444;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Trình sinh tiêu đề thông minh</h1>", unsafe_allow_html=True)

# User sẽ input ở đây
text_input = st.text_area("Nhập đoạn văn của bạn:", height=200)

# Nơi sinh tiêu đề nè
if st.button("Sinh tiêu đề"):
    if text_input.strip():
        # Preprocess input
        text_input = word_tokenize(text_input, format="text")
        
        inputs = tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in inputs.items()}

        # Generate title
        outputs = model.generate(
            inputs["input_ids"],
            max_length=80,
            num_beams=4,
            early_stopping=True
        )
        title = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Hiển thị tiêu đề ở đây
        st.markdown(f"<h2 style='text-align: center; color: #cccccc;'>{title}</h2>", unsafe_allow_html=True)
    else:
        st.warning("Vui lòng nhập văn bản trước khi nhấn sinh tiêu đề.")

st.markdown("---")
st.caption("Made by My group ✨")
