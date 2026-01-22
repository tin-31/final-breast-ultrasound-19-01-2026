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

# 🔥 FILE ID (Giữ nguyên)
SEG_FILE_ID = '1eUtmSEXAh9r-o_qRSk5oaYK7yfxjITfl' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'TRUST_MED_SEG_MODEL.pth'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

# --- 1. TẢI VÀ LOAD MODEL ---
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

# --- 2. CÁC HÀM XỬ LÝ (CẬP NHẬT MỚI) ---

def clean_mask_output(mask_prob, threshold=0.5):
    """
    Hàm hậu xử lý: Lọc nhiễu muối tiêu và chỉ giữ lại khối u lớn nhất.
    """
    # 1. Thresholding
    mask_binary = (mask_prob > threshold).astype(np.uint8)
    
    # 2. Morphological Opening (Xóa đốm trắng nhỏ)
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    
    # 3. Giữ lại vùng lớn nhất (Largest Component Analysis)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(mask_clean) # Không tìm thấy gì trả về đen
    
    # Tìm contour lớn nhất
    c = max(contours, key=cv2.contourArea)
    
    # Tạo mask mới chỉ chứa contour lớn nhất này
    final_mask = np.zeros_like(mask_clean)
    cv2.drawContours(final_mask, [c], -1, 1, thickness=cv2.FILLED)
    
    return final_mask

def get_bounding_box_from_mask(mask_clean, padding=0.2):
    """Tìm tọa độ từ mask đã làm sạch"""
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        
        # Logic Padding
        pad_w = int(rw * padding); pad_h = int(rh * padding)
        x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
        x2 = min(mask_clean.shape[1], x + rw + pad_w)
        y2 = min(mask_clean.shape[0], y + rh + pad_h)
        return (x1, y1, x2, y2), "Soft-ROI"
    else:
        # Fallback
        h, w = mask_clean.shape
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
st.title("🩺 TRUST-MED: Chẩn đoán Ung thư Vú Đa trung tâm")
st.markdown("---")

uploaded_file = st.file_uploader("Tải ảnh siêu âm (JPG/PNG):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    with st.spinner("🤖 AI đang phân tích (Đã bật bộ lọc nhiễu)..."):
        # 1. Segmentation Inference
        preprocess_seg = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_seg = preprocess_seg(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mask_logits = seg_model(input_seg)
            mask_prob = torch.sigmoid(mask_logits).cpu().detach().numpy()[0,0]
        
        # Resize xác suất về kích thước gốc
        mask_prob_resized = cv2.resize(mask_prob, (img_np.shape[1], img_np.shape[0]))
        
        # --- 🔥 QUAN TRỌNG: HẬU XỬ LÝ LÀM SẠCH MASK ---
        # Tăng ngưỡng lên 0.6 để lọc bớt nhiễu mờ
        mask_clean = clean_mask_output(mask_prob_resized, threshold=0.6) 
        
        # Tính tỷ lệ diện tích u
        mask_ratio = np.sum(mask_clean) / (img_np.shape[0]*img_np.shape[1])
        
        # Tạo ảnh hiển thị mask (Trắng/Đen)
        mask_display = cv2.cvtColor(mask_clean * 255, cv2.COLOR_GRAY2RGB)
        if mask_ratio == 0:
            cv2.putText(mask_display, "No Tumor Detected", (50, img_np.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 2. Cắt ảnh dựa trên Mask sạch
        (x1, y1, x2, y2), roi_type = get_bounding_box_from_mask(mask_clean)
        roi_img = img_np[y1:y2, x1:x2]
        
        # 3. Vẽ Khung Đỏ
        img_with_box = img_np.copy()
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 4) 
        label_box = "TUMOR DETECTED" if roi_type == "Soft-ROI" else "FALLBACK AREA"
        cv2.putText(img_with_box, label_box, (x1, max(y1-10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 4. Classification
        roi_pil = Image.fromarray(roi_img)
        trans_cls = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_cls = trans_cls(roi_pil).unsqueeze(0).to(DEVICE)
        
        heatmap, pred_idx, probs = cam_extractor(input_cls)
        trust_score = calc_trust_score(probs, mask_ratio)
        
        # 5. Logic Kết quả
        probs_np = probs.detach().numpy()[0]
        prob_benign = probs_np[0] + probs_np[1]
        prob_malignant = probs_np[2] + probs_np[3]
        
        prob_normal =
