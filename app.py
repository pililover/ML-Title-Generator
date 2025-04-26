import streamlit as st
from dotenv import load_dotenv
import os

from underthesea import word_tokenize
from transformers import EncoderDecoderModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

st.set_page_config(page_title="Trình sinh tiêu đề", layout="centered")

# Config
torch.classes.__path__ = []  # add this line to manually set it to empty.
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# List model
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

SUMMARIZATION_MODELS = {}

# Load pre-trained model và tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path, model_type, use_auth_token=False):
    try:
        token_arg = HUGGINGFACE_TOKEN if use_auth_token and HUGGINGFACE_TOKEN else None

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

        # Load the appropriate model type
        if model_type == "encoder-decoder":
            model = EncoderDecoderModel.from_pretrained(model_path, use_auth_token=token_arg)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=token_arg)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model and tokenizer loaded successfully for {model_path}.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Lỗi tải mô hình '{model_path}' hoặc tokenizer '{tokenizer_path}': {e}")
        print(f"Error loading model/tokenizer: {e}")
        return None, None

# Khởi tạo lịch sử trong session_state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False
if "selected_history_index" not in st.session_state:
    st.session_state.selected_history_index = None

# Sidebar cho History
with st.sidebar:
    if st.button("🧾 Hiện/Ẩn lịch sử"):
        st.session_state.show_sidebar = not st.session_state.show_sidebar

if st.session_state.show_sidebar:
    with st.sidebar:
        st.markdown("### 🕓 Lịch sử")
        if not st.session_state.history:
            st.write("Chưa có lịch sử nào.")
        else:
            for idx, history_item in enumerate(st.session_state.history):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"- {history_item['title']}", key=f"history_{idx}"):
                        st.session_state.selected_history_index = idx
                        # Cập nhật text input và hiển thị kết quả cũ
                        st.session_state.text_input_area = history_item["input_text"]
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        st.session_state.history.pop(idx)
                        if st.session_state.selected_history_index == idx:
                            st.session_state.selected_history_index = None
                        st.rerun()

# Một chút styling
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
            width: 100%; /* Make button full width */
            margin-top: 10px;
        }
        .stButton > button:hover {
            background-color: #444444;
        }
        div[role="radiogroup"] label {
            margin-right: 15px;
        }
        /* Style history buttons */
        button[kind="secondary"] {
            background-color: transparent !important;
            color: white !important;
            text-align: left !important;
            width: 100% !important;
        }
        button[kind="secondary"]:hover {
            background-color: #2a2a2a !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main app
st.markdown("<h1 style='text-align: center; color: white;'>Trình sinh tiêu đề thông minh</h1>", unsafe_allow_html=True)

# 1. Chọn task
task_option = st.radio(
    "Chọn chức năng bạn muốn:",
    ('Sinh tiêu đề', 'Tóm tắt nội dung'),
    horizontal=True,
    key="task_selection"
)

# 2. Chọn model
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

# 3. Text input
text_input = st.text_area("Nhập đoạn văn của bạn:", height=200, key="text_input_area")

# Hiển thị nội dung lịch sử đã chọn
if st.session_state.selected_history_index is not None:
    selected_history = st.session_state.history[st.session_state.selected_history_index]
    st.markdown("---")
    st.markdown(f"<h3 style='color: #cccccc;'>Tiêu đề đã tạo trước đó:</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white; background-color: #2a2a2a; padding: 10px; border-radius: 5px;'>{selected_history['title']}</p>", unsafe_allow_html=True)

# Tạo nút và xử lí logic
button_label = f"{task_option}"
if st.button(button_label, key="generate_button"):
    if not model_config:
        st.warning("Vui lòng chọn mô hình.")
    elif not text_input.strip():
        st.warning("Vui lòng nhập văn bản trước khi thực hiện.")
    else:
        # Load model được chọn và tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_config["model_path"],
            model_config["tokenizer_path"],
            model_config["model_type"],
            model_config.get("use_auth_token", False)
        )

        if model and tokenizer:
            # Tiền xử lí input - underthesea
            processed_text = word_tokenize(text_input, format="text")

            try:
                # Tokenization
                inputs = tokenizer(
                    processed_text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Generation
                progress_message = st.empty()
                progress_message.info(f"Đang {task_option.lower()} với mô hình: '{selected_model_key}'...")

                with torch.no_grad():
                    if task_option == 'Sinh tiêu đề':
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=80,
                            num_beams=5,
                            early_stopping=True,
                            no_repeat_ngram_size=2
                        )
                    else:
                        st.error("Tác vụ không xác định.")
                        outputs = None

                if outputs is not None:
                    # Decode và hiện kết quả
                    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                    # Lưu vào lịch sử
                    st.session_state.history.append({
                        "title": result,
                        "input_text": text_input
                    })
                    st.session_state.selected_history_index = len(st.session_state.history) - 1

                    # Hiển thị kết quả
                    st.markdown("---")
                    result_label = "Tiêu đề được tạo:" if task_option == 'Sinh tiêu đề' else "Nội dung tóm tắt:"
                    st.markdown(f"<h3 style='color: #cccccc;'>{result_label}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: white; background-color: #2a2a2a; padding: 10px; border-radius: 5px;'>{result}</p>", unsafe_allow_html=True)

                progress_message.empty()

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                print(f"Error during processing: {e}")
# if st.button("Sinh tiêu đề"):
#     if text_input.strip():
#         # Preprocess input
#         text_input = word_tokenize(text_input, format="text")
        
#         inputs = tokenizer(
#             text_input,
#             padding="max_length",
#             truncation=True,
#             max_length=256,
#             return_tensors="pt"
#         )
#         inputs = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in inputs.items()}

#         # Generate title
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_length=80,
#             num_beams=4,
#             early_stopping=True
#         )
#         title = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

#         # Hiển thị tiêu đề ở đây
#         st.markdown(f"<h2 style='text-align: center; color: #cccccc;'>{title}</h2>", unsafe_allow_html=True)
#     else:
#         st.warning("Vui lòng nhập văn bản trước khi nhấn sinh tiêu đề.")

# st.markdown("---")
# st.caption("Made by My group ✨")