import streamlit as st

st.set_page_config(page_title="Trình sinh tiêu đề", layout="centered")

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

text_input = st.text_area("Nhập đoạn văn của bạn:", height=200)

if st.button("Sinh tiêu đề"):
    if text_input.strip():
        title = "generate_title(text_input)"  # Giả lập
        st.markdown(f"<h2 style='text-align: center; color: #cccccc;'>{title}</h2>", unsafe_allow_html=True)
    else:
        st.warning("Vui lòng nhập văn bản trước khi nhấn sinh tiêu đề.")

st.markdown("---")
st.caption("Made by My group ✨")
