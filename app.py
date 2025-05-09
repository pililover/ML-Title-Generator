import streamlit as st
import os
import time
from datetime import datetime
from underthesea import word_tokenize
from transformers import EncoderDecoderModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import transformers
import google.generativeai as genai
from utils.preprocessing import clean_text, segment_text
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
# Giảm bớt cảnh báo
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# Cấu hình Streamlit
st.set_page_config(page_title="Trình sinh tiêu đề", layout="centered")

# Cấu hình Gemini API (thay YOUR_GEMINI_API_KEY bằng API key thực tế)
GEMINI_API_KEY = "AIzaSyCEDRquPDC9N09hTHGD9FfvsPP83AZT78Q"  # Thay bằng API key thực tế của bạn
genai.configure(api_key=GEMINI_API_KEY)

# Các mô hình
TITLE_MODELS = {
    "PhoBERT Encoder-Decoder": {
        "model_path": "PuppetLover/Title_generator",
        "tokenizer_path": "vinai/phobert-base-v2",
        "token": True,
        "model_type": "encoder-decoder"
    },
    "ViT5 Title Generator": {
        "model_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "tokenizer_path": "HTThuanHcmus/vit5-base-vietnews-summarization-finetune",
        "token": False,
        "model_type": "seq2seq"
    },
    "BARTpho Title Generator": {
        "model_path": "HTThuanHcmus/bartpho-finetune",
        "tokenizer_path": "HTThuanHcmus/bartpho-finetune",
        "token": False,
        "model_type": "seq2seq"
    },
    "Gemini Title Generator": {
        # "model_path": "gemini-1.5-pro",
        "model_path" : "gemini-1.5-flash",
        "tokenizer_path": None,
        "token": False,
        "model_type": "gemini"
    }
}

SUMMARIZATION_MODELS = {
    "ViT5 Summarization": {
        "model_path": "HTThuanHcmus/vit5-summarization-news-finetune",
        "tokenizer_path": "HTThuanHcmus/vit5-summarization-news-finetune",
        "token": False,
        "model_type": "seq2seq"
    },
    "BARTpho Summarization": {
        "model_path": "HTThuanHcmus/bartpho-summarization-news-finetune",
        "tokenizer_path": "HTThuanHcmus/bartpho-summarization-news-finetune",
        "token": False,
        "model_type": "seq2seq"
    },
    "Gemini Summarization": {
        "model_path": "gemini-1.5-pro",
        "tokenizer_path": None,
        "token": False,
        "model_type": "gemini"
    }
}

# Cache load model/tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path, model_type, token=False):
    if model_type == "gemini":
        model = genai.GenerativeModel(model_path)
        return model, None
    token_arg = None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False) if tokenizer_path else None
    if model_type == "encoder-decoder":
        model = EncoderDecoderModel.from_pretrained(model_path, token=token_arg)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=token_arg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

# Hàm xử lý Gemini
def generate_with_gemini(model, text, task):
    prompt = (
        f"Với tư cách một chuyên gia hãy tạo tiêu đề ngắn gọn cho văn bản sau: {text}" if task == "Sinh tiêu đề"
        else f"Vơi tư cách một chuyên gia hãy tạo tóm tắt cho văn bản: {text}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()

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
                    # Rút gọn câu đầu để hiển thị
                    short_preview = history_item['title'].split('.')[0][:60]
                    if len(history_item['title']) > 60:
                        short_preview += "..."
                    if st.button(f"- {short_preview}", key=f"history_{idx}"):
                        st.session_state.selected_history_index = idx
                        st.session_state.current_generated = None
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        st.session_state.history.pop(idx)
                        if st.session_state.selected_history_index == idx:
                            st.session_state.selected_history_index = None
                        st.rerun()


# Một chút CSS
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        textarea {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            font-family: 'Courier New', monospace;
            border: 1px solid #ffffff30 !important;
            border-radius: 10px !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            margin-top: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            transform: scale(1.02);
        }
        div[role="radiogroup"] label {
            margin-right: 15px;
            background-color: #2c2f36;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        div[role="radiogroup"] input:checked + label {
            background-color: #0078FF;
            color: white;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .card {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Main App
st.markdown("""
    <h1 style='text-align: center; color: white; font-family: "Segoe UI", sans-serif;'>
        Trình Sinh Tiêu Đề & Tóm Tắt
    </h1>
""", unsafe_allow_html=True)

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
            model_config.get("token", False)
        )

        if model:
            if model_config["model_type"] == "gemini":
                processed_text = clean_text(text_input)
                try:
                    with st.spinner(f"⏳ Đang {task_option.lower()} với mô hình '{selected_model_key}'..."):
                        result = generate_with_gemini(model, processed_text, task_option)
                    
                    st.session_state.current_generated = result
                    st.session_state.current_task = task_option

                    st.session_state.history.append({
                        "title": result,
                        "input_text": text_input,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name": selected_model_key
                    })
                    st.session_state.selected_history_index = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi với Gemini: {e}")
                    print(f"Error during Gemini processing: {e}")
            else:
                if model_config["model_type"] == "encoder-decoder":
                    processed_text = clean_text(text_input)
                    processed_text = segment_text(processed_text)
                else:
                    processed_text = clean_text(text_input)
                    

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

                    with st.spinner(f"⏳ Đang {task_option.lower()} với mô hình '{selected_model_key}'..."):
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_length=80 if task_option == 'Sinh tiêu đề' else 200,
                                num_beams=5,
                                early_stopping=True,
                                no_repeat_ngram_size=2
                            )
                        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                        result = result.replace("_", " ")

                    st.session_state.current_generated = result
                    st.session_state.current_task = task_option

                    st.session_state.history.append({
                        "title": result,
                        "input_text": text_input,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name": selected_model_key
                    })
                    st.session_state.selected_history_index = None
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
