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
DEVICE = 'cpu' # Chạy trên Streamlit Cloud dùng CPU

# 🔥 FILE ID (Đã cập nhật từ link bạn gửi)
SEG_FILE_ID = '1eUtmSEXAh9r-o_qRSk5oaYK7yfxjITfl' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'TRUST_MED_SEG_MODEL.pth'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

# --- 1. TẢI VÀ LOAD MODEL ---
@st.cache_resource
def load_models():
    # Tải file từ Drive nếu chưa có (Chạy 1 lần duy nhất)
    if not os.path.exists(SEG_PATH):
        url = f'https://drive.google.com/uc?id={SEG_FILE_ID}'
        gdown.download(url, SEG_PATH, quiet=False)
    
    if not os.path.exists(CLS_PATH):
        url = f'https://drive.google.com/uc?id={CLS_FILE_ID}'
        gdown.download(url, CLS_PATH, quiet=False)

    # Load Segmentation Model
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, decoder_attention_type="scse")
    seg_model.load_state_dict(torch.load(SEG_PATH, map_location=torch.device(DEVICE)))
    seg_model.eval()
    
    # Load Classification Model
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location=torch.device(DEVICE)))
    cls_model.eval()
    
    return seg_model, cls_model

try:
    with st.spinner("⏳ Đang tải dữ liệu từ đám mây (Lần đầu mất khoảng 1 phút)..."):
        seg_model, cls_model = load_models()
except Exception as e:
    st.error(f"Lỗi khởi động: {e}. Vui lòng kiểm tra lại kết nối mạng.")
    st.stop()

# --- 2. CÁC HÀM XỬ LÝ ẢNH ---
def get_bounding_box(mask_pred, padding=0.2):
    """Tìm tọa độ khối u để vẽ và cắt"""
    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        # Mở rộng vùng (Padding)
        pad_w = int(rw * padding); pad_h = int(rh * padding)
        x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
        x2 = min(mask_pred.shape[1], x + rw + pad_w)
        y2 = min(mask_pred.shape[0], y + rh + pad_h)
        return (x1, y1, x2, y2), "Soft-ROI"
    else:
        # Fallback trung tâm
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
st.title("🩺 TRUST-MED: Hệ thống Chẩn đoán Ung thư Vú Đa trung tâm")
st.markdown("---")

col_upload, col_info = st.columns([1, 2])
with col_upload:
    uploaded_file = st.file_uploader("Tải ảnh siêu âm:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- XỬ LÝ ẢNH ---
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    with st.spinner("🤖 AI đang phân tích toàn diện..."):
        # 1. Phân đoạn (Segmentation)
        preprocess_seg = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_seg = preprocess_seg(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_logits = seg_model(input_seg)
            mask_pred = (torch.sigmoid(mask_logits) > 0.5).float().numpy()[0,0]
        
        # Resize mask về gốc & Tính tỷ lệ diện tích
        mask_resized = cv2.resize(mask_pred, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_ratio = np.sum(mask_resized) / (img_np.shape[0]*img_np.shape[1])
        
        # 2. Lấy tọa độ Box & Cắt ROI
        (x1, y1, x2, y2), roi_type = get_bounding_box(mask_resized.astype(np.uint8))
        roi_img = img_np[y1:y2, x1:x2]
        
        # 3. Vẽ Bounding Box lên ảnh gốc (Khoanh vùng)
        img_with_box = img_np.copy()
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá
        cv2.putText(img_with_box, "AI Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Phân loại & Grad-CAM
        roi_pil = Image.fromarray(roi_img)
        trans_cls = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_cls = trans_cls(roi_pil).unsqueeze(0).to(DEVICE)
        
        heatmap, pred_idx, probs = cam_extractor(input_cls)
        trust_score = calc_trust_score(probs, mask_ratio)
        
        # 5. Tính nhóm xác suất Lành/Ác
        probs_np = probs.detach().numpy()[0]
        prob_benign = probs_np[0] + probs_np[1] # BR2 + BR3
        prob_malignant = probs_np[2] + probs_np[3] # BR4A + BR4B+
        
        # Xử lý trường hợp "Bình thường" (Không có u)
        prob_normal = 0.0
        if mask_ratio < 0.005: # Nếu diện tích u < 0.5% ảnh
            prob_normal = 0.95
            prob_benign = 0.05
            prob_malignant = 0.0
            status_text = "Bình thường (Không phát hiện u)"
            status_color = "green"
        else:
            if prob_malignant > prob_benign:
                status_text = "Nghi ngờ ÁC TÍNH"
                status_color = "red"
            else:
                status_text = "Khả năng cao LÀNH TÍNH"
                status_color = "blue"

    # --- HIỂN THỊ KẾT QUẢ ---
    # Hàng 1: Hình ảnh trực quan
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_with_box, caption="Khoanh vùng tổn thương (Detection)", use_column_width=True)
    
    with col2:
        # Overlay Grad-CAM
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(cv2.resize(roi_img, (224,224)), 0.6, heatmap_colored, 0.4, 0)
        st.image(superimposed, caption="Vùng AI 'soi' (Grad-CAM)", use_column_width=True)
        
    with col3:
        st.subheader("📊 Kết quả phân tích")
        st.markdown(f"Chẩn đoán: **:{status_color}[{status_text}]**")
        if prob_normal < 0.5:
            st.markdown(f"Chi tiết: **BI-RADS {['2', '3', '4A', '4B+'][pred_idx]}**")
        st.markdown(f"Độ tin cậy: **{trust_score:.1%}**")
        
        st.markdown("---")
        st.write("Xác suất bệnh học:")
        if prob_normal > 0.5:
             st.progress(int(prob_normal * 100), text=f"Mô bình thường: {prob_normal:.1%}")
        else:
            st.progress(int(prob_benign * 100), text=f"Lành tính / Theo dõi: {prob_benign:.1%}")
            st.progress(int(prob_malignant * 100), text=f"Ác tính (Nguy cơ cao): {prob_malignant:.1%}")
        

    # Cảnh báo cuối
    if trust_score < 0.4 and prob_normal < 0.5:
        st.warning("⚠️ CẢNH BÁO: Độ tin cậy thấp. Vui lòng kiểm tra lại góc chụp hoặc tham vấn bác sĩ.")
    elif pred_idx == 3 and prob_normal < 0.5:
        st.error("🚨 KHUYẾN NGHỊ: Cần thực hiện sinh thiết ngay để xác định khối u.")
    elif prob_normal > 0.5:
        st.success("✅ Không phát hiện dấu hiệu bất thường.")

    with st.expander("🔍 Xem thông số kỹ thuật (Debug Info)"):
        st.json({
            "Probabilities": {
                "BI-RADS 2": f"{probs_np[0]:.4f}",
                "BI-RADS 3": f"{probs_np[1]:.4f}",
                "BI-RADS 4A": f"{probs_np[2]:.4f}",
                "BI-RADS 4B+": f"{probs_np[3]:.4f}"
            },
            "Segmentation Info": {
                "Tumor Area Ratio": f"{mask_ratio:.4f}",
                "ROI Mode": roi_type
            }
        })
