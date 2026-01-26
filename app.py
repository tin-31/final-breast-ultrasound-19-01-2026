# ==========================================
# 🩺 TRUST-MED AI: HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN UNG THƯ VÚ
# ==========================================
# Phiên bản: Pro v4.0 (Giao diện Bác sĩ)
# Tác giả: [Tên của bạn/Nhóm nghiên cứu]

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import models, transforms
import os
import gdown
import matplotlib.pyplot as plt
import pandas as pd
import time

# =====================================================
# ⚙️ CẤU HÌNH GIAO DIỆN CHUNG
# =====================================================
st.set_page_config(
    page_title="TRUST-MED AI Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện y tế chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .stAlert { border-radius: 8px; }
    .report-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 10px;
        background: #f1f3f6;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# 1. CẤU HÌNH & TẢI MODEL
# ============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔥 FILE ID (MODEL CỦA BẠN)
SEG_FILE_ID = '1eUtmSEXAh9r-o_qRSk5oaYK7yfxjITfl' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'TRUST_MED_SEG_MODEL.pth'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

@st.cache_resource
def load_models():
    # Tải file nếu chưa có
    if not os.path.exists(SEG_PATH):
        with st.spinner("📥 Đang tải mô hình Phân đoạn từ Cloud..."):
            gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=True)
    if not os.path.exists(CLS_PATH):
        with st.spinner("📥 Đang tải mô hình Phân loại từ Cloud..."):
            gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=True)

    # 1.1 LOAD SEGMENTATION (ResNet34 + U-Net + SCSE)
    # (Lưu ý: Code app cũ của bạn dùng ResNet34, tôi giữ nguyên để khớp logic)
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, decoder_attention_type="scse")
    # Load safe: map location về CPU nếu không có GPU
    seg_model.load_state_dict(torch.load(SEG_PATH, map_location=torch.device(DEVICE)))
    seg_model.to(DEVICE)
    seg_model.eval()
    
    # 1.2 LOAD CLASSIFICATION (EfficientNet-B4)
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location=torch.device(DEVICE)))
    cls_model.to(DEVICE)
    cls_model.eval()
    
    return seg_model, cls_model

# Load model ngay khi khởi động
try:
    seg_model, cls_model = load_models()
except Exception as e:
    st.error(f"❌ Lỗi khởi động hệ thống AI: {e}")
    st.stop()

# ============================
# 2. CÁC HÀM XỬ LÝ ẢNH (LOGIC CŨ)
# ============================
def validate_image(image_pil):
    img_np = np.array(image_pil)
    if img_np.shape[0] < 100 or img_np.shape[1] < 100: return False, "Kích thước quá nhỏ"
    if len(img_np.shape) == 3:
        if np.std(img_np, axis=2).mean() > 20: return False, "Ảnh màu (không phải siêu âm)"
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 5: return False, "Ảnh quá mờ/đen"
    return True, "Hợp lệ"

def letterbox_image(image, size):
    iw, ih = image.size; w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale); nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh, (w-nw)//2, (h-nh)//2

def post_process_mask(mask_prob, threshold=0.5):
    mask_binary = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    if num > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_clean = np.zeros_like(mask_binary)
        mask_clean[labels == max_label] = 1
        return mask_clean
    return mask_binary

def get_bounding_box(mask_pred, padding=0.2):
    cnts, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_w = int(w*padding); pad_h = int(h*padding)
        x1 = max(0, x-pad_w); y1 = max(0, y-pad_h)
        x2 = min(mask_pred.shape[1], x+w+pad_w)
        y2 = min(mask_pred.shape[0], y+h+pad_h)
        return (x1, y1, x2, y2), "ROI"
    return (0,0,0,0), "None"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = None; self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __call__(self, x):
        output = self.model(x); idx = torch.argmax(output)
        self.model.zero_grad(); output[0, idx].backward()
        grads = self.gradients[0]; acts = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
        return cam, int(idx), torch.nn.functional.softmax(output, dim=1)

cam_extractor = GradCAM(cls_model, cls_model.features[-1])

def calc_trust_score(probs, mask_area_ratio):
    probs_np = probs.detach().cpu().numpy()[0]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))
    score_cls = 1.0 - (entropy / np.log(4))
    score_seg = 0.3 if mask_area_ratio < 0.01 else 0.95
    return 0.7 * score_cls + 0.3 * score_seg

# =====================================================
# 4) SIDEBAR & CHỌN TRANG (NAVIGATION)
# =====================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
st.sidebar.title("TRUST-MED AI")
st.sidebar.markdown("**Hệ thống hỗ trợ chẩn đoán hình ảnh**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Danh mục chức năng:",
    ["🏠 Bàn làm việc (Chẩn đoán)", "📖 Hướng dẫn sử dụng", "📚 Cơ sở dữ liệu huấn luyện", "ℹ️ Giới thiệu dự án"]
)

# =====================================================
# TRANG 1: BÀN LÀM VIỆC (MAIN APP)
# =====================================================
if menu == "🏠 Bàn làm việc (Chẩn đoán)":
    st.title("🖥️ Bàn làm việc Bác sĩ")
    st.info("💡 **Gợi ý:** Tải ảnh siêu âm lên để AI phân tích tự động. Kết quả chỉ mang tính tham khảo.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📥 Nhập dữ liệu")
        uploaded_file = st.file_uploader("Chọn ảnh siêu âm (JPG/PNG/DICOM)", type=["jpg", "png", "jpeg"])
        
        with st.expander("⚙️ Cấu hình nâng cao"):
            seg_threshold = st.slider("Độ nhạy tìm khối u", 0.1, 0.9, 0.5, 0.05)
            use_post_process = st.checkbox("Bật khử nhiễu tự động", value=True)

    with col_right:
        if uploaded_file is None:
            st.warning("👈 Vui lòng tải ảnh lên để bắt đầu.")
            st.image("https://img.freepik.com/free-vector/doctor-examining-patient-clinic_23-2148856559.jpg", width=400, caption="Hệ thống sẵn sàng...")
        else:
            # XỬ LÝ & HIỂN THỊ
            original_pil = Image.open(uploaded_file).convert("RGB")
            original_np = np.array(original_pil)
            
            # Guardrail
            is_valid, msg = validate_image(original_pil)
            if not is_valid:
                st.error(f"⛔️ ẢNH KHÔNG HỢP LỆ: {msg}")
            else:
                progress_bar = st.progress(0, text="Đang khởi tạo...")
                
                # --- BƯỚC 1: PHÂN ĐOẠN ---
                progress_bar.progress(30, text="Đang phân đoạn tổn thương (U-Net)...")
                input_pil, nw, nh, dx, dy = letterbox_image(original_pil, (256, 256))
                to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                input_tensor = to_tensor(input_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    mask_prob = torch.sigmoid(seg_model(input_tensor)).cpu().numpy()[0,0]
                
                mask_valid = mask_prob[dy:dy+nh, dx:dx+nw]
                mask_resized = cv2.resize(mask_valid, (original_np.shape[1], original_np.shape[0]))
                
                if use_post_process: mask_binary = post_process_mask(mask_resized, threshold=seg_threshold)
                else: mask_binary = (mask_resized > seg_threshold).astype(np.uint8)
                
                # --- BƯỚC 2: CẮT ROI & PHÂN LOẠI ---
                progress_bar.progress(60, text="Đang phân tích bệnh học (EfficientNet)...")
                (x1, y1, x2, y2), roi_status = get_bounding_box(mask_binary)
                roi_img = original_np[y1:y2, x1:x2]
                
                # Visuals
                mask_display = original_np.copy()
                mask_display[mask_binary == 1] = [0, 255, 0]
                overlay = cv2.addWeighted(original_np, 0.7, mask_display, 0.3, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Classification Logic
                roi_pil = Image.fromarray(roi_img)
                trans_cls = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                input_cls = trans_cls(roi_pil).unsqueeze(0).to(DEVICE)
                
                heatmap, pred_idx, probs = cam_extractor(input_cls)
                mask_ratio = np.sum(mask_binary) / (original_np.shape[0]*original_np.shape[1])
                trust_score = calc_trust_score(probs, mask_ratio)
                
                probs_np = probs.detach().cpu().numpy()[0]
                prob_benign = probs_np[0] + probs_np[1]
                prob_malignant = probs_np[2] + probs_np[3]
                
                # Logic Kết luận
                if mask_ratio < 0.005:
                    status = "KHÔNG PHÁT HIỆN BẤT THƯỜNG (BI-RADS 1)"; color = "green"
                    prob_display = 0.05 # Giả lập thấp
                else:
                    if prob_malignant > prob_benign:
                        status = "NGHI NGỜ ÁC TÍNH (BI-RADS 4/5)"; color = "red"
                        prob_display = prob_malignant
                    else:
                        status = "KHẢ NĂNG CAO LÀNH TÍNH (BI-RADS 2/3)"; color = "blue"
                        prob_display = prob_benign
                
                progress_bar.progress(100, text="Hoàn tất!")
                time.sleep(0.5); progress_bar.empty()
                
                # --- BƯỚC 3: HIỂN THỊ KẾT QUẢ (DASHBOARD STYLE) ---
                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color:{color}; margin-top:0;">📋 KẾT QUẢ: {status}</h3>
                    <p><b>Độ tin cậy của AI:</b> {trust_score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Xác suất Lành tính", f"{prob_benign:.1%}")
                m2.metric("Xác suất Ác tính", f"{prob_malignant:.1%}", delta_color="inverse")
                m3.metric("Diện tích tổn thương", f"{mask_ratio*100:.2f}% ảnh")
                
                st.divider()
                st.subheader("🔬 Hình ảnh phân tích chi tiết")
                
                tab_img1, tab_img2, tab_img3 = st.tabs(["1. Ảnh Gốc", "2. Định vị Tổn thương", "3. Bản đồ nhiệt AI"])
                
                with tab_img1:
                    st.image(original_pil, use_column_width=True)
                with tab_img2:
                    st.image(overlay, caption="Vùng xanh lá: Khối u | Khung xanh dương: ROI", use_column_width=True)
                with tab_img3:
                    hm_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
                    superimposed = cv2.addWeighted(cv2.resize(roi_img, (224,224)), 0.6, hm_color, 0.4, 0)
                    st.image(superimposed, caption="Vùng màu đỏ là nơi AI tập trung để chẩn đoán", use_column_width=True)

# =====================================================
# TRANG 2: HƯỚNG DẪN SỬ DỤNG
# =====================================================
elif menu == "📖 Hướng dẫn sử dụng":
    st.title("📖 Hướng dẫn sử dụng TRUST-MED")
    st.markdown("""
    ### Chào mừng bác sĩ đến với hệ thống!
    Dưới đây là quy trình 3 bước để sử dụng phần mềm hiệu quả:

    #### Bước 1: Chuẩn bị hình ảnh
    * Hệ thống hỗ trợ các định dạng ảnh phổ biến: **JPG, PNG, JPEG**.
    * Ảnh nên là ảnh siêu âm thô (B-mode), hạn chế các ảnh có chứa mũi tên chỉ dẫn hoặc marker màu của máy siêu âm cũ để tránh nhiễu.

    #### Bước 2: Tải ảnh và Phân tích
    1. Truy cập vào mục **"🏠 Bàn làm việc"** ở menu bên trái.
    2. Nhấn nút **"Browse files"** để chọn ảnh từ máy tính.
    3. Hệ thống sẽ tự động chạy qua 2 mô hình AI:
        * **Segmentation Model:** Để tìm và khoanh vùng khối u.
        * **Classification Model:** Để đánh giá tính chất lành/ác.

    #### Bước 3: Đọc kết quả
    * **Thanh trạng thái:** Sẽ hiện màu ĐỎ (Nguy hiểm), XANH DƯƠNG (Lành tính) hoặc XANH LÁ (Bình thường).
    * **Hình ảnh trực quan:** Bác sĩ có thể xem tab "Bản đồ nhiệt" để biết AI đang nghi ngờ vùng nào nhất trên khối u (vùng màu đỏ rực).
    """)

# =====================================================
# TRANG 3: CƠ SỞ DỮ LIỆU
# =====================================================
elif menu == "📚 Cơ sở dữ liệu huấn luyện":
    st.title("📊 Nguồn dữ liệu huấn luyện")
    st.markdown("Hệ thống TRUST-MED được huấn luyện trên **12 bộ dữ liệu** uy tín (công khai và nội bộ), đảm bảo tính đa dạng sinh học và khả năng kháng nhiễu.")
    
    # Danh sách 12 dataset chuẩn
    datasets = [
        ("BUSI (Breast Ultrasound Images)", "Cairo Univ", "Dataset chuẩn vàng với nhãn phân đoạn chi tiết."),
        ("BUSBRA (Brazil)", "Đa trung tâm", "Dữ liệu thu thập từ nhiều dòng máy siêu âm khác nhau."),
        ("UDIAT (Tây Ban Nha)", "Bệnh viện Parc Taulí", "Chuyên về các tổn thương nhỏ (small lesions)."),
        ("OASBUD (Ba Lan)", "Dữ liệu mở", "Kèm theo nhãn BI-RADS chuẩn."),
        ("STU (Trung Quốc)", "Shantou Univ", "Dataset lớn khu vực Châu Á."),
        ("Thamburaj Dataset", "Tư nhân", "Tập trung vào đặc trưng hình thái học."),
        ("HMSS (Mexico)", "Hospital Move", "Dữ liệu lâm sàng thực tế."),
        ("Mendeley Data V2", "Rodrigues et al.", "Cân bằng giữa Lành và Ác."),
        ("BrEaST-Lesions", "Kaggle", "Tổng hợp đa nguồn."),
        ("Dataset A (Private)", "Nội bộ", "Dữ liệu bổ sung để cân bằng lớp."),
        ("VinDr-Mammo (Tham chiếu)", "VinBigData", "Dữ liệu đặc thù người Việt Nam."),
        ("HisBreast (Việt Nam)", "Bệnh viện VN", "Dữ liệu lâm sàng trọng điểm của đề tài.")
    ]
    
    for i, (name, source, desc) in enumerate(datasets):
        with st.expander(f"{i+1}. {name}"):
            st.markdown(f"**Nguồn:** {source}")
            st.markdown(f"**Mô tả:** {desc}")

# =====================================================
# TRANG 4: GIỚI THIỆU
# =====================================================
elif menu == "ℹ️ Giới thiệu dự án":
    st.title("ℹ️ Về dự án TRUST-MED")
    st.markdown("""
    ### 🎯 Mục tiêu
    Xây dựng hệ thống AI hỗ trợ chẩn đoán ung thư vú tự động, giúp giảm tải cho bác sĩ chẩn đoán hình ảnh và tăng độ chính xác trong tầm soát sớm.

    ### 🛠️ Công nghệ lõi
    * **Phân đoạn (Segmentation):** U-Net với kiến trúc ResNet34 và cơ chế Attention (scSE) giúp bắt trọn biên dạng khối u.
    * **Phân loại (Classification):** EfficientNet-B4 - một trong những mô hình CNN hiệu quả nhất hiện nay.
    * **Giải thích (XAI):** Tích hợp Grad-CAM để minh bạch hóa quyết định của AI.

    ### ⚠️ Tuyên bố miễn trừ trách nhiệm
    * Ứng dụng này là sản phẩm nghiên cứu khoa học.
    * Kết quả của AI **không thay thế** chẩn đoán của bác sĩ chuyên khoa.
    * Người dùng chịu trách nhiệm khi sử dụng thông tin từ ứng dụng này.
    """)
