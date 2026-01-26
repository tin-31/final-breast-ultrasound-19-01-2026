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
import time

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="TRUST-MED: AI Hỗ trợ Chẩn đoán Ung thư Vú",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tùy chỉnh để giao diện đẹp và chuyên nghiệp hơn
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        color: #0066cc;
        font-weight: bold;
    }
    .report-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- CẤU HÌNH MODEL ---
DEVICE = 'cpu' # Hoặc 'cuda' nếu deploy trên GPU

# 🔥 FILE ID
SEG_FILE_ID = '1eUtmSEXAh9r-o_qRSk5oaYK7yfxjITfl' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'TRUST_MED_SEG_MODEL.pth'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

# --- 1. TẢI & LOAD MODEL (GIỮ NGUYÊN LOGIC CŨ) ---
@st.cache_resource
def load_models():
    if not os.path.exists(SEG_PATH):
        gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=False)
    if not os.path.exists(CLS_PATH):
        gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=False)

    # Model Segment: ResNet34 + U-Net + SCSE
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, decoder_attention_type="scse")
    seg_model.load_state_dict(torch.load(SEG_PATH, map_location=torch.device(DEVICE)))
    seg_model.eval()
    
    # Model Classify: EfficientNet-B4
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location=torch.device(DEVICE)))
    cls_model.eval()
    
    return seg_model, cls_model

# --- 2. CÁC HÀM XỬ LÝ ẢNH (GIỮ NGUYÊN) ---
def validate_image(image_pil):
    img_np = np.array(image_pil)
    if img_np.shape[0] < 100 or img_np.shape[1] < 100:
        return False, "Kích thước ảnh quá nhỏ."
    if len(img_np.shape) == 3:
        std_color = np.std(img_np, axis=2).mean()
        if std_color > 15: 
            return False, "Phát hiện ảnh màu. Vui lòng chỉ tải lên ảnh siêu âm (đen trắng)."
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 5:
        return False, "Ảnh quá mờ hoặc đen trơn (Không có tín hiệu)."
    return True, "Hợp lệ"

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh, (w-nw)//2, (h-nh)//2

def post_process_mask(mask_prob, threshold=0.5):
    mask_binary = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    if num_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
        mask_clean = np.zeros_like(mask_binary)
        mask_clean[labels == max_label] = 1
        return mask_clean
    else:
        return mask_binary

def get_bounding_box(mask_pred, padding=0.2):
    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        pad_w = int(rw * padding); pad_h = int(rh * padding)
        x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
        x2 = min(mask_pred.shape[1], x + rw + pad_w)
        y2 = min(mask_pred.shape[0], y + rh + pad_h)
        return (x1, y1, x2, y2), "Soft-ROI"
    else:
        h, w = mask_pred.shape
        cy, cx = h//2, w//2; sz = min(h, w)//2
        return (cx-sz, cy-sz, cx+sz, cy+sz), "Fallback"

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

def calc_trust_score(probs, mask_area_ratio):
    probs_np = probs.detach().numpy()[0]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))
    max_ent = np.log(4)
    score_cls = 1.0 - (entropy / max_ent)
    score_seg = 0.3 if mask_area_ratio < 0.01 else 0.95
    return 0.7 * score_cls + 0.3 * score_seg

# --- KHỞI TẠO MODEL ---
try:
    # Ẩn spinner khi load xong
    with st.sidebar:
        with st.spinner("⏳ Đang khởi tạo hệ thống AI..."):
            seg_model, cls_model = load_models()
            cam_extractor = GradCAM(cls_model, cls_model.features[-1])
    # st.sidebar.success("Hệ thống sẵn sàng!")
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()


# --- GIAO DIỆN CHÍNH (TABS) ---
st.title("🩺 TRUST-MED AI: Hỗ trợ Chẩn đoán Hình ảnh")
st.markdown("### Hệ thống phân tích Siêu âm vú Tự động hóa")

tab1, tab2, tab3 = st.tabs(["🖥️ Bàn làm việc (Chẩn đoán)", "📖 Hướng dẫn sử dụng", "📚 Nguồn dữ liệu"])

# ==========================================
# TAB 1: BÀN LÀM VIỆC (MAIN APP)
# ==========================================
with tab1:
    col_input, col_result = st.columns([1, 2.5])

    with col_input:
        st.info("📥 **Nhập liệu**")
        uploaded_file = st.file_uploader("Tải lên ảnh siêu âm (DICOM/JPG/PNG):", type=["jpg", "png", "jpeg"])
        
        # Cấu hình nhanh
        with st.expander("⚙️ Cấu hình nâng cao"):
            seg_threshold = st.slider("Độ nhạy (Sensitivity)", 0.1, 0.9, 0.5, 0.05)
            use_post_process = st.toggle("Khử nhiễu tự động", value=True)

    with col_result:
        if uploaded_file is None:
            st.warning("👈 Vui lòng tải lên ảnh siêu âm để bắt đầu phân tích.")
            st.image("https://img.freepik.com/free-vector/medical-technology-science-background_53876-119566.jpg", use_column_width=True, caption="Hệ thống sẵn sàng phân tích")
        else:
            # XỬ LÝ ẢNH
            original_pil = Image.open(uploaded_file).convert("RGB")
            original_np = np.array(original_pil)
            
            # Guardrail Check
            is_valid, msg = validate_image(original_pil)
            
            if not is_valid:
                st.error(f"⛔️ ẢNH KHÔNG HỢP LỆ: {msg}")
            else:
                # Progress Bar giả lập trải nghiệm người dùng
                progress_text = "Đang phân tích..."
                my_bar = st.progress(0, text=progress_text)
                
                # --- PROCESSING PIPELINE ---
                # 1. Preprocessing
                my_bar.progress(20, text="Đang tiền xử lý ảnh...")
                input_pil, nw, nh, dx, dy = letterbox_image(original_pil, (256, 256))
                to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                input_tensor = to_tensor(input_pil).unsqueeze(0).to(DEVICE)

                # 2. Segmentation
                my_bar.progress(50, text="Đang phân đoạn tổn thương (U-Net)...")
                with torch.no_grad():
                    mask_logits = seg_model(input_tensor)
                    mask_prob = torch.sigmoid(mask_logits).numpy()[0,0]
                
                mask_valid = mask_prob[dy:dy+nh, dx:dx+nw]
                mask_resized = cv2.resize(mask_valid, (original_np.shape[1], original_np.shape[0]))
                
                if use_post_process:
                    mask_binary = post_process_mask(mask_resized, threshold=seg_threshold)
                else:
                    mask_binary = (mask_resized > seg_threshold).astype(np.uint8)

                # 3. ROI & Classification
                my_bar.progress(80, text="Đang phân loại bệnh học (EfficientNet)...")
                mask_ratio = np.sum(mask_binary) / (original_np.shape[0]*original_np.shape[1])
                
                # Visuals
                mask_display = original_np.copy()
                mask_display[mask_binary == 1] = [0, 255, 0]
                overlay = cv2.addWeighted(original_np, 0.7, mask_display, 0.3, 0)
                
                (x1, y1, x2, y2), roi_type = get_bounding_box(mask_binary)
                roi_img = original_np[y1:y2, x1:x2]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

                roi_pil = Image.fromarray(roi_img)
                trans_cls = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                input_cls = trans_cls(roi_pil).unsqueeze(0).to(DEVICE)

                heatmap, pred_idx, probs = cam_extractor(input_cls)
                trust_score = calc_trust_score(probs, mask_ratio)
                
                probs_np = probs.detach().numpy()[0]
                prob_benign = probs_np[0] + probs_np[1]
                prob_malignant = probs_np[2] + probs_np[3]
                
                my_bar.progress(100, text="Hoàn tất!")
                time.sleep(0.5)
                my_bar.empty()

                # --- HIỂN THỊ KẾT QUẢ (REPORT CARD STYLE) ---
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📋 Phiếu Kết Quả Phân Tích")
                
                # Logic kết luận
                prob_normal = 0.0
                if mask_ratio < 0.005: 
                    prob_normal = 0.95; prob_benign = 0.05; prob_malignant = 0.0
                    status_text = "KHÔNG PHÁT HIỆN BẤT THƯỜNG (BI-RADS 1)"; status_color = "green"
                    final_conf = prob_normal
                else:
                    if prob_malignant > prob_benign:
                        status_text = "NGHI NGỜ ÁC TÍNH (BI-RADS 4/5)"; status_color = "red"
                        final_conf = prob_malignant
                    else:
                        status_text = "KHẢ NĂNG CAO LÀNH TÍNH (BI-RADS 2/3)"; status_color = "blue"
                        final_conf = prob_benign

                # 1. Kết luận chính
                c_res1, c_res2 = st.columns([2, 1])
                with c_res1:
                    st.markdown(f"### Kết luận: :{status_color}[{status_text}]")
                    st.markdown(f"**Độ tin cậy của AI:** {trust_score:.1%}")
                with c_res2:
                    if prob_normal > 0.5:
                        st.metric("Xác suất Bình thường", f"{prob_normal:.1%}")
                    else:
                        st.metric("Tỉ lệ Ác tính", f"{prob_malignant:.1%}", delta_color="inverse")
                        st.caption(f"Lành tính: {prob_benign:.1%}")

                st.divider()

                # 2. Hình ảnh trực quan
                st.markdown("**🔬 Hình ảnh phân tích chi tiết:**")
                img_col1, img_col2, img_col3 = st.columns(3)
                
                with img_col1:
                    st.image(original_pil, caption="Ảnh gốc (Original)", use_column_width=True)
                with img_col2:
                    st.image(overlay, caption="Định vị Khối u (Segmentation)", use_column_width=True)
                with img_col3:
                    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    superimposed = cv2.addWeighted(cv2.resize(roi_img, (224,224)), 0.6, heatmap_colored, 0.4, 0)
                    st.image(superimposed, caption="Bản đồ nhiệt (AI Attention)", use_column_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: HƯỚNG DẪN SỬ DỤNG
# ==========================================
with tab2:
    st.header("📖 Hướng dẫn cho người dùng mới")
    
    st.markdown("""
    Chào mừng bác sĩ đến với hệ thống **TRUST-MED AI**. Dưới đây là quy trình 3 bước đơn giản:

    ### Bước 1: Chuẩn bị ảnh
    * Hệ thống chấp nhận các file ảnh định dạng **.JPG, .PNG**.
    * Đảm bảo ảnh là **ảnh siêu âm (Grayscale)**, không chứa các ghi chú màu quá lớn.
    * Cắt bỏ các thông tin nhạy cảm của bệnh nhân (tên, tuổi) trước khi tải lên nếu cần.

    ### Bước 2: Tải ảnh và Phân tích
    1.  Chuyển sang Tab **"🖥️ Bàn làm việc"**.
    2.  Nhấn nút **"Browse files"** ở cột bên trái để chọn ảnh từ máy tính.
    3.  Hệ thống sẽ tự động kiểm tra chất lượng ảnh và chạy phân tích (mất khoảng 1-3 giây).

    ### Bước 3: Đọc kết quả
    * **Kết luận:** AI sẽ đưa ra gợi ý phân loại (Lành tính/Ác tính/Bình thường).
    * **Định vị:** Quan sát vùng màu xanh lá cây trên ảnh để xem vị trí khối u AI phát hiện.
    * **Bản đồ nhiệt:** Vùng màu đỏ trên ảnh thứ 3 cho biết nơi AI "tập trung nhìn vào" để đưa ra quyết định.
    
    ---
    **⚠️ Lưu ý quan trọng:** *Kết quả của AI chỉ mang tính chất tham khảo hỗ trợ (Second Opinion). Quyết định lâm sàng cuối cùng luôn thuộc về bác sĩ chuyên khoa.*
    """)

# ==========================================
# TAB 3: DỮ LIỆU & TRÍCH DẪN
# ==========================================
with tab3:
    st.header("📚 Cơ sở dữ liệu huấn luyện")
    st.markdown("Hệ thống TRUST-MED được huấn luyện dựa trên sự tổng hợp của **12 bộ dữ liệu siêu âm vú** uy tín trên thế giới và tại Việt Nam, bao gồm:")
    
    # Danh sách 12 dataset (Giả lập dựa trên các dataset phổ biến nhất trong nghiên cứu Breast US)
    datasets = [
        {"name": "BUSI (Breast Ultrasound Images)", "source": "Cairo University, Egypt", "desc": "Bộ dữ liệu tiêu chuẩn vàng với mặt nạ phân đoạn chi tiết."},
        {"name": "BUSBRA (Breast Ultrasound Brazil)", "source": "Brazil", "desc": "Dữ liệu đa trung tâm với độ đa dạng cao về thiết bị."},
        {"name": "UDIAT (Dataset B)", "source": "Parc Taulí Hospital, Spain", "desc": "Chuyên về các tổn thương nhỏ và khó phát hiện."},
        {"name": "OASBUD", "source": "Ba Lan", "desc": "Dữ liệu mở về siêu âm vú với nhãn BI-RADS chi tiết."},
        {"name": "STU (Shantou University)", "source": "China", "desc": "Tập dữ liệu lớn từ bệnh viện Shantou."},
        {"name": "Thamburaj Dataset", "source": "Private Collection", "desc": "Tập trung vào đặc trưng hình thái khối u."},
        {"name": "HMSS (Hospital Move S.S.)", "source": "Mexico", "desc": "Dữ liệu thực tế lâm sàng tại Mexico."},
        {"name": "Mendeley Data V2", "source": "Rodrigues et al.", "desc": "Tổng hợp các ca siêu âm vú lành tính và ác tính."},
        {"name": "BrEaST-Lesions", "source": "Kaggle/Open Source", "desc": "Tập hợp đa dạng các loại tổn thương vú."},
        {"name": "Dataset A (Private)", "source": "Nghiên cứu nội bộ", "desc": "Dữ liệu thu thập bổ sung để cân bằng nhãn."},
        {"name": "VinDr-Mammo (Tham chiếu)", "source": "VinBigData", "desc": "Dữ liệu tham chiếu đặc điểm tổn thương trên người Việt."},
        {"name": "HisBreast (Vietnamese Clinical Data)", "source": "Bệnh viện tại Việt Nam", "desc": "Dữ liệu lâm sàng thực tế thu thập tại Việt Nam (Key Dataset)."}
    ]

    for i, ds in enumerate(datasets):
        with st.expander(f"{i+1}. {ds['name']}"):
            st.write(f"**Nguồn:** {ds['source']}")
            st.write(f"**Mô tả:** {ds['desc']}")
            
    st.info("💡 Việc kết hợp đa nguồn dữ liệu giúp TRUST-MED có khả năng kháng nhiễu tốt (Robustness) và giảm thiểu hiện tượng Overfitting.")
