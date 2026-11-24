import streamlit as st
import json
import os
import glob
import time
from openai import OpenAI
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS

st.set_page_config(page_title="MedAgentBoard - Phòng Hội Chẩn AI", page_icon="🏥", layout="wide")

st.title("🏥 MedAgentBoard: Mô Phỏng Hội Chẩn Đa Tác Nhân (Multi-Agent MDT)")
st.markdown("Quan sát trực tiếp quá trình các bác sĩ AI phân tích, tổng hợp và phản biện ca bệnh y khoa.")

# Sidebar cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình phòng hội chẩn")
    available_models = [k for k in LLM_MODELS_SETTINGS.keys() if LLM_MODELS_SETTINGS[k].get("api_key")]
    default_idx = available_models.index("gemini-3.1-flash-lite") if "gemini-3.1-flash-lite" in available_models else 0
    selected_model = st.selectbox("Chọn mô hình LLM:", available_models, index=default_idx)
    
    st.divider()
    mode = st.radio("Chế độ hiển thị:", ["🔴 Hội chẩn trực tiếp (Live MDT)", "📂 Xem lại lịch sử từ thư mục Logs"])

# Hàm gọi LLM dùng chung cấu hình dự án
def call_agent(model_key, system_prompt, user_prompt):
    cfg = LLM_MODELS_SETTINGS[model_key]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=30.0)
    response = client.chat.completions.create(
        model=cfg["model_name"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ==========================================
# CHẾ ĐỘ 1: HỘI CHẨN TRỰC TIẾP (LIVE MDT)
# ==========================================
if mode == "🔴 Hội chẩn trực tiếp (Live MDT)":
    # Tải sẵn câu hỏi mẫu từ PubMedQA nếu có
    sample_questions = {}
    data_path = "./my_datasets/processed/medqa/PubMedQA/medqa_mc_test.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            items = json.load(f)
            for item in items:
                sample_questions[f"{item['qid']} (Đáp án gốc: {item['answer']})"] = item

    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        st.subheader("📋 Chọn hoặc nhập ca bệnh")
        if sample_questions:
            chosen_sample = st.selectbox("Chọn nhanh từ PubMedQA:", ["Tự nhập câu hỏi mới"] + list(sample_questions.keys()))
        else:
            chosen_sample = "Tự nhập câu hỏi mới"

        if chosen_sample != "Tự nhập câu hỏi mới":
            q_data = sample_questions[chosen_sample]
            question_text = st.text_area("Nội dung câu hỏi & bối cảnh:", value=q_data["question"], height=220)
            options_text = st.text_input("Các lựa chọn:", value=str(q_data["options"]))
        else:
            question_text = st.text_area("Nội dung câu hỏi & bối cảnh:", value="Bệnh nhân nam 58 tuổi đau ngực trái lan lên vai kèm vã mồ hôi 30 phút. Điện tâm đồ có ST chênh lên ở V1-V4. Hướng xử trí ưu tiên hàng đầu là gì?", height=220)
            options_text = st.text_input("Các lựa chọn:", value="{'A': 'Tái tưới máu mạch vành khẩn cấp', 'B': 'Cho về theo dõi ngoại trú', 'C': 'Chỉ định nội soi dạ dày'}")

        start_btn = st.button("🚀 Bắt đầu phiên hội chẩn", type="primary", use_container_width=True)

    with col_input2:
        st.subheader("💬 Diễn biến phòng họp đa chuyên khoa")
        chat_container = st.container()

        if start_btn:
            specialties = [
                ("Doctor 1", "Nội khoa (Internal Medicine)", "🩺"),
                ("Doctor 2", "Ngoại khoa (Surgery)", "🔪"),
                ("Doctor 3", "Chẩn đoán hình ảnh (Radiology)", "🩻")
            ]
            
            # Sơ đồ trạng thái trực quan
            status_cols = st.columns(4)
            s_doc1 = status_cols[0].empty()
            s_doc2 = status_cols[1].empty()
            s_doc3 = status_cols[2].empty()
            s_meta = status_cols[3].empty()
            status_boxes = [s_doc1, s_doc2, s_doc3]

            for i, (_, name, icon) in enumerate(specialties):
                status_boxes[i].info(f"{icon} {name}\n\n⏳ Đang chờ...")
            s_meta.info("👨‍⚕️ Bác sĩ trưởng\n\n⏳ Đang chờ...")

            doctor_opinions = []

            # Vòng 1: Các bác sĩ chuyên khoa phân tích độc lập
            with chat_container:
                st.markdown("### 🔄 Vòng 1: Phân tích độc lập từ các chuyên khoa")
                for i, (doc_id, spec_name, icon) in enumerate(specialties):
                    status_boxes[i].warning(f"{icon} {spec_name}\n\n🧠 Đang phân tích...")
                    with st.chat_message(doc_id, avatar=icon):
                        st.markdown(f"**{doc_id} - Chuyên khoa {spec_name}**")
                        with st.spinner("Đang suy luận y khoa..."):
                            sys_p = f"Bạn là bác sĩ chuyên khoa {spec_name} tham gia hội chẩn đa chuyên khoa (MDT). Hãy phân tích ngắn gọn, súc tích bằng tiếng Việt (kèm thuật ngữ y khoa) và chọn đáp án phù hợp nhất."
                            usr_p = f"Câu hỏi: {question_text}\nLựa chọn: {options_text}\nHãy đưa ra lập luận chuyên môn và đáp án đề xuất của bạn."
                            reply = call_agent(selected_model, sys_p, usr_p)
                            st.write(reply)
                            doctor_opinions.append(f"{spec_name}: {reply}")
                    status_boxes[i].success(f"{icon} {spec_name}\n\n✅ Đã phát biểu")

                # Bác sĩ trưởng tổng hợp
                st.markdown("### 👨‍⚕️ Tổng hợp & Điều phối đồng thuận")
                s_meta.warning("👨‍⚕️ Bác sĩ trưởng\n\n🧠 Đang tổng hợp...")
                with st.chat_message("Meta", avatar="👨‍⚕️"):
                    st.markdown("**Bác sĩ trưởng (Meta Agent - Điều phối viên)**")
                    with st.spinner("Đang tổng hợp ý kiến các chuyên khoa..."):
                        meta_sys = "Bạn là Bác sĩ trưởng điều phối hội chẩn. Hãy tổng hợp ý kiến từ 3 chuyên khoa, chỉ ra điểm đồng thuận/bất đồng và đề xuất kết luận chung."
                        meta_usr = f"Ca bệnh: {question_text}\nLựa chọn: {options_text}\n\nÝ kiến các bác sĩ:\n" + "\n---\n".join(doctor_opinions)
                        synthesis = call_agent(selected_model, meta_sys, meta_usr)
                        st.write(synthesis)
                s_meta.success("👨‍⚕️ Bác sĩ trưởng\n\n✅ Đã tổng hợp")

                # Vòng 2: Các bác sĩ phản hồi đồng thuận
                st.markdown("### 🤝 Vòng 2: Phản biện & Xác nhận đồng thuận")
                review_cols = st.columns(3)
                for i, (doc_id, spec_name, icon) in enumerate(specialties):
                    with review_cols[i]:
                        with st.spinner(f"{spec_name} đang xem xét..."):
                            rev_sys = f"Bạn là bác sĩ {spec_name}. Hãy xem bản tổng hợp của Bác sĩ trưởng và trả lời ngắn gọn: bạn ĐỒNG Ý hay KHÔNG ĐỒNG Ý, nêu rõ lý do trong 2 câu."
                            rev_usr = f"Bản tổng hợp:\n{synthesis}\n\nÝ kiến phản hồi của bạn:"
                            rev_reply = call_agent(selected_model, rev_sys, rev_usr)
                            st.success(f"**{icon} {spec_name}**\n\n{rev_reply}")

# ==========================================
# CHẾ ĐỘ 2: XEM LẠI LOGS TỪ THỰC NGHIỆM
# ==========================================
else:
    st.subheader("📂 Trình xem lại phiên hội chẩn đã chạy trong `logs/`")
    log_files = sorted(glob.glob("logs/**/*.json", recursive=True), key=os.path.getmtime, reverse=True)
    
    if not log_files:
        st.warning("Chưa tìm thấy file kết quả `.json` nào trong thư mục `logs/`.")
    else:
        selected_log = st.selectbox("Chọn file log kết quả:", log_files)
        with open(selected_log, "r") as f:
            log_data = json.load(f)
        
        st.json(log_data)
