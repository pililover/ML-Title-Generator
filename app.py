import streamlit as st
from dotenv import load_dotenv
import os
import time
from datetime import datetime

from underthesea import word_tokenize
from transformers import EncoderDecoderModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import transformers

# Giảm bớt cảnh báo
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# Cấu hình Streamlit
st.set_page_config(page_title="Trình sinh tiêu đề", layout="centered")

# Load biến môi trường
torch.classes.__path__ = []
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Các mô hình
TITLE_MODELS = {
    "PhoBERT Encoder-Decoder": {
        "model_path": "PuppetLover/Title_generator",
        "tokenizer_path": "PuppetLover/Finetune_PhoBert",
        "use_auth_token": True,
        "model_type": "encoder-decoder"
    },
    "ViT5 Title Generator": {
        "model_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "tokenizer_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "use_auth_token": False,
        "model_type": "seq2seq"
    },
    "BARTpho Title Generator": {
        "model_path": "HTThuanHcmus/bartpho-finetune",
        "tokenizer_path": "HTThuanHcmus/bartpho-finetune",
        "use_auth_token": False,
        "model_type": "seq2seq"
    }
}

SUMMARIZATION_MODELS = {
    "ViT5 Summarization": {
        "model_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "tokenizer_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "use_auth_token": False,
        "model_type": "seq2seq"
    }
}

# Cache load model/tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path, model_type, use_auth_token=False):
    token_arg = HUGGINGFACE_TOKEN if use_auth_token and HUGGINGFACE_TOKEN else None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if model_type == "encoder-decoder":
        model = EncoderDecoderModel.from_pretrained(model_path, use_auth_token=token_arg)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=token_arg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

# Init session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False
if "selected_history_index" not in st.session_state:
    st.session_state.selected_history_index = None
if "current_generated" not in st.session_state:
    st.session_state.current_generated = None
if "current_task" not in st.session_state:
    st.session_state.current_task = None

# Sidebar
with st.sidebar:
    if st.button("🧾 Hiện/Ẩn lịch sử"):
        st.session_state.show_sidebar = not st.session_state.show_sidebar

if st.session_state.show_sidebar:
    with st.sidebar:
        st.markdown("### 🕓 Lịch sử")
        if not st.session_state.history:
            st.write("Chưa có lịch sử nào.")
        else:
            if st.button("🗑️ Xóa tất cả lịch sử"):
                st.session_state.history = []
                st.session_state.selected_history_index = None
                st.rerun()

            for idx, history_item in enumerate(st.session_state.history):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"- {history_item['title']}", key=f"history_{idx}"):
                        st.session_state.selected_history_index = idx
                        st.session_state.current_generated = None
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        st.session_state.history.pop(idx)
                        if st.session_state.selected_history_index == idx:
                            st.session_state.selected_history_index = None
                        st.rerun()

# Một chút css
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
            width: 100%;
            margin-top: 10px;
        }
        .stButton > button:hover {
            background-color: #444444;
        }
        div[role="radiogroup"] label {
            margin-right: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Main App
st.markdown("<h1 style='text-align: center; color: white;'>Trình sinh tiêu đề và tóm tắt</h1>", unsafe_allow_html=True)

task_option = st.radio(
    "Chọn chức năng bạn muốn:",
    ('Sinh tiêu đề', 'Tóm tắt nội dung'),
    horizontal=True,
    key="task_selection"
)

selected_model_key = None
model_config = None

if task_option == 'Sinh tiêu đề':
    selected_model_key = st.selectbox(
        "Chọn mô hình sinh tiêu đề:",
        list(TITLE_MODELS.keys()),
        key="title_model_selector"
    )
    model_config = TITLE_MODELS[selected_model_key]

elif task_option == 'Tóm tắt nội dung':
    selected_model_key = st.selectbox(
        "Chọn mô hình tóm tắt:",
        list(SUMMARIZATION_MODELS.keys()),
        key="summary_model_selector"
    )
    model_config = SUMMARIZATION_MODELS[selected_model_key]

# Upload file
uploaded_file = st.file_uploader("Hoặc tải lên file (.txt, .docx):", type=["txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    if file_name.endswith(".txt"):
        text_input = uploaded_file.read().decode("utf-8")
    elif file_name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        text_input = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    st.text_area("Nội dung file đã tải lên:", value=text_input, height=200, key="text_input_area", disabled=True)
else:
    text_input = st.text_area("Nhập đoạn văn của bạn:", height=200, key="text_input_area")

# Nút bấm sau phần nhập văn bản
button_label = f"{task_option}"
if st.button(button_label, key="generate_button"):
    if not model_config:
        st.warning("Vui lòng chọn mô hình.")
    elif not text_input.strip():
        st.warning("Vui lòng nhập văn bản hoặc tải file lên.")
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_config["model_path"],
            model_config["tokenizer_path"],
            model_config["model_type"],
            model_config.get("use_auth_token", False)
        )

        if model and tokenizer:
            processed_text = word_tokenize(text_input, format="text")

            try:
                inputs = tokenizer(
                    processed_text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {key: value.to(device) for key, value in inputs.items()}

                progress_message = st.empty()
                progress_message.info(f"Đang {task_option.lower()} với mô hình: '{selected_model_key}'...")

                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=80 if task_option == 'Sinh tiêu đề' else 200,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )

                result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                st.session_state.current_generated = result
                st.session_state.current_task = task_option

                st.session_state.history.append({
                    "title": result,
                    "input_text": text_input,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": selected_model_key
                })
                st.session_state.selected_history_index = None

                progress_message.empty()

                st.rerun()

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                print(f"Error during processing: {e}")

# Hiển thị kết quả sinh mới
if st.session_state.current_generated:
    st.markdown("---")
    label_text = "Tiêu đề được tạo:" if st.session_state.current_task == 'Sinh tiêu đề' else "Nội dung tóm tắt:"
    st.markdown(f"<h3 style='color: #cccccc;'>{label_text}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white; background-color: #2a2a2a; padding: 10px; border-radius: 5px;'>"
                f"{st.session_state.current_generated}</p>", unsafe_allow_html=True)

# Hiển thị lịch sử
if st.session_state.selected_history_index is not None and st.session_state.selected_history_index < len(st.session_state.history):
    selected_history = st.session_state.history[st.session_state.selected_history_index]
    st.markdown("---")
    st.markdown(f"<h3 style='color: #cccccc;'>Kết quả đã tạo:</h3>", unsafe_allow_html=True)

    if f"show_full_input_{st.session_state.selected_history_index}" not in st.session_state:
        st.session_state[f"show_full_input_{st.session_state.selected_history_index}"] = False

    show_full = st.session_state[f"show_full_input_{st.session_state.selected_history_index}"]

    input_text_to_display = selected_history['input_text'] if show_full else (selected_history['input_text'][:1000] + "..." if len(selected_history['input_text']) > 1000 else selected_history['input_text'])

    st.markdown(f"""
    <div style='color: white; background-color: #2a2a2a; padding: 10px; border-radius: 5px;'>
        <b>Model:</b> {selected_history['model_name']}<br>
        <b>Thời gian:</b> {selected_history['timestamp']}<br><br>
        <b>Văn bản gốc:</b><br>
        <div style='background-color: #3a3a3a; padding: 8px; border-radius: 5px; margin-bottom: 10px;'>{input_text_to_display}</div>
    """, unsafe_allow_html=True)

    if len(selected_history['input_text']) > 1000:
        if not show_full:
            if st.button("📖 Xem đầy đủ văn bản", key=f"show_full_{st.session_state.selected_history_index}"):
                st.session_state[f"show_full_input_{st.session_state.selected_history_index}"] = True
                st.rerun()
        else:
            if st.button("🔽 Thu gọn văn bản", key=f"collapse_full_{st.session_state.selected_history_index}"):
                st.session_state[f"show_full_input_{st.session_state.selected_history_index}"] = False
                st.rerun()

    st.markdown(f"""
        <b>Kết quả:</b><br>
        <div style='background-color: #3a3a3a; padding: 8px; border-radius: 5px;'>{selected_history['title']}</div>
    </div>
    """, unsafe_allow_html=True)
