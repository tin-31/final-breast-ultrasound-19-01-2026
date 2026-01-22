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

# --- CẤU HÌNH ---
st.set_page_config(page_title="TRUST-MED AI", page_icon="🩺", layout="wide")
DEVICE = 'cpu'

# 🔥 FILE ID (Đã đúng, giữ nguyên)
SEG_FILE_ID = '1eUtmSEXAh9r-o_qRSk5oaYK7yfxjITfl' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'TRUST_MED_SEG_MODEL.pth'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

# --- 1. TẢI & LOAD MODEL ---
@st.cache_resource
def load_models():
    if not os.path.exists(SEG_PATH):
        gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=False)
    if not os.path.exists(CLS_PATH):
        gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=False)

    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, decoder_attention_type="scse")
    seg_model.load_state_dict(torch.load(SEG_PATH, map_location=torch.device(DEVICE)))
    seg_model.eval()
    
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location=torch.device(DEVICE)))
    cls_model.eval()
    
    return seg_model, cls_model

try:
    with st.spinner("⏳ Đang khởi động hệ thống..."):
        seg_model, cls_model = load_models()
except Exception as e:
    st.error(f"Lỗi khởi động: {e}")
    st.stop()

# --- 2. HÀM XỬ LÝ ẢNH THÔNG MINH (LETTERBOX) ---
def letterbox_image(image, size):
    '''Resize ảnh giữ nguyên tỷ lệ bằng cách thêm viền đen'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0)) # Viền đen
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh, (w-nw)//2, (h-nh)//2

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

cam_extractor = GradCAM(cls_model, cls_model.features[-1])

def calc_trust_score(probs, mask_area_ratio):
    probs_np = probs.detach().numpy()[0]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))
    max_ent = np.log(4)
    score_cls = 1.0 - (entropy / max_ent)
    score_seg = 0.3 if mask_area_ratio < 0.01 else 0.95
    return 0.7 * score_cls + 0.3 * score_seg

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🩺 TRUST-MED: Chẩn đoán Ung thư Vú (Pro)")

# --- SIDEBAR TÙY CHỈNH (CỨU TINH DEBUG) ---
with st.sidebar:
    st.header("⚙️ Cấu hình nâng cao")
    seg_threshold = st.slider("Độ nhạy tìm u (Threshold)", 0.1, 0.9, 0.5, 0.05, 
                              help="Giảm xuống nếu không thấy mask, tăng lên nếu mask bị nhiễu.")
    show_debug = st.checkbox("Chế độ Debug (Xem ảnh input)")

uploaded_file = st.file_uploader("Tải ảnh siêu âm:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_pil = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_pil)
    
    with st.spinner("🤖 AI đang xử lý..."):
        # 1. PREPROCESS THÔNG MINH (Letterbox)
        # Resize ảnh về 256x256 nhưng giữ tỷ lệ, thêm viền đen
        input_pil, nw, nh, dx, dy = letterbox_image(original_pil, (256, 256))
        
        # Debug: Xem ảnh AI nhìn thấy
        if show_debug:
            st.sidebar.image(input_pil, caption="Input vào Model (256x256)")
        
        # Chuyển sang Tensor
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = to_tensor(input_pil).unsqueeze(0).to(DEVICE)

        # 2. SEGMENTATION
        with torch.no_grad():
            mask_logits = seg_model(input_tensor)
            mask_prob = torch.sigmoid(mask_logits).numpy()[0,0]
        
        # 3. POST-PROCESS MASK (Gỡ bỏ viền đen)
        # Cắt bỏ phần viền đen đã thêm vào lúc Letterbox
        mask_valid = mask_prob[dy:dy+nh, dx:dx+nw]
        # Resize về kích thước ảnh gốc
        mask_resized = cv2.resize(mask_valid, (original_np.shape[1], original_np.shape[0]))
        
        # Áp dụng Threshold từ thanh trượt
        mask_binary = (mask_resized > seg_threshold).astype(np.uint8)
        
        # Tạo Mask hiển thị (Xanh lá mờ)
        mask_display = original_np.copy()
        # Tìm vùng mask = 1 để tô màu
        mask_indices = mask_binary == 1
        mask_display[mask_indices] = [0, 255, 0] # Tô xanh lá pixel u
        # Trộn với ảnh gốc: 0.7 ảnh gốc + 0.3 màu xanh
        overlay = cv2.addWeighted(original_np, 0.7, mask_display, 0.3, 0)

        # Kiểm tra mask rỗng
        mask_ratio = np.sum(mask_binary) / (original_np.shape[0]*original_np.shape[1])
        if mask_ratio == 0:
            cv2.putText(overlay, "No Tumor Detected", (50, original_np.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 4. GET BOX & CROP
        (x1, y1, x2, y2), roi_type = get_bounding_box(mask_binary)
        roi_img = original_np[y1:y2, x1:x2]
        
        # Vẽ khung đỏ lên ảnh overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        
        # 5. CLASSIFICATION & GRAD-CAM
        roi_pil = Image.fromarray(roi_img)
        trans_cls = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_cls = trans_cls(roi_pil).unsqueeze(0).to(DEVICE)
        
        heatmap, pred_idx, probs = cam_extractor(input_cls)
        trust_score = calc_trust_score(probs, mask_ratio)
        
        # Logic Kết quả
        probs_np = probs.detach().numpy()[0]
        prob_benign = probs_np[0] + probs_np[1]
        prob_malignant = probs_np[2] + probs_np[3]
        
        prob_normal = 0.0
        if mask_ratio < 0.005: 
            prob_normal = 0.95; prob_benign = 0.05; prob_malignant = 0.0
            status_text = "Bình thường"; status_color = "green"
        else:
            if prob_malignant > prob_benign:
                status_text = "Nghi ngờ ÁC TÍNH"; status_color = "red"
            else:
                status_text = "Khả năng cao LÀNH TÍNH"; status_color = "blue"

    # --- HIỂN THỊ 3 CỘT (TỐI ƯU HÓA) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(overlay, caption=f"1. Phân đoạn & Định vị (Ngưỡng: {seg_threshold})", use_column_width=True)
    
    with col2:
        st.image(roi_img, caption="2. Ảnh cắt (ROI)", use_column_width=True)
        
    with col3:
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(cv2.resize(roi_img, (224,224)), 0.6, heatmap_colored, 0.4, 0)
        st.image(superimposed, caption="3. AI 'soi' (Grad-CAM)", use_column_width=True)

    # --- BẢNG KẾT QUẢ ---
    st.divider()
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader(f":{status_color}[{status_text}]")
        if prob_normal < 0.5:
            st.write(f"Chi tiết: **BI-RADS {['2', '3', '4A', '4B+'][pred_idx]}**")
        st.metric("Độ tin cậy (TRUST-Score)", f"{trust_score:.1%}")
        
    with c2:
        st.write("**Phân tích xác suất:**")
        if prob_normal > 0.5:
             st.progress(int(prob_normal * 100), text=f"Mô bình thường: {prob_normal:.1%}")
        else:
            st.progress(int(prob_benign * 100), text=f"Lành tính / Theo dõi: {prob_benign:.1%}")
            st.progress(int(prob_malignant * 100), text=f"Ác tính (Nguy cơ cao): {prob_malignant:.1%}")

    if trust_score < 0.4 and prob_normal < 0.5:
        st.warning("⚠️ Nếu Mask chưa chuẩn, hãy điều chỉnh thanh trượt bên trái (Sidebar).")
