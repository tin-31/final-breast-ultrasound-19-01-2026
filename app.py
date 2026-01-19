import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
import os
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="TRUST-MED: AI Siêu âm Vú", page_icon="🎗️", layout="wide")

# ID Google Drive
SEG_FILE_ID = '1WYMlDSjXCnPE21C2jy6jRk7NlZGEnaCm' 
CLS_FILE_ID = '1P2nuQ9HbJliaItRP-F9oeVmcDhYZ4Ju8'
SEG_PATH = 'TRUST_MED_FINAL_ROBUST.pth'
CLS_PATH = 'TRUST_MED_BIRADS_EXPERT_FINETUNED.pth'
DEVICE = 'cpu'

# --- 2. KIẾN TRÚC MODEL & GRAD-CAM HOOK ---
class BiradsNet_FineTune(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder EfficientNet-B0
        self.encoder = smp.encoders.get_encoder(name="efficientnet-b0", in_channels=3, depth=5, weights=None)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        out_channels = self.encoder.out_channels[-1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )
        # Biến để lưu gradient phục vụ Grad-CAM
        self.gradients = None
    
    # Hàm hook để bắt gradient
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        features_list = self.encoder(x)
        x = features_list[-1] # Feature map cuối cùng (7x7 hoặc 8x8)
        
        # Đăng ký hook nếu đang cần tính Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        x = self.avgpool(x)
        x = self.head(x)
        return x, features_list[-1] # Trả về cả feature map để tính CAM

# --- 3. HÀM TÍNH GRAD-CAM (QUAN TRỌNG) ---
def generate_gradcam(model, img_tensor, target_class_index):
    model.eval()
    
    # 1. Forward pass
    # Bật gradient để tính đạo hàm ngược
    img_tensor.requires_grad = True
    output, feature_map = model(img_tensor)
    
    # 2. Zero gradients
    model.zero_grad()
    
    # 3. Backward pass cho class mục tiêu
    # (Ví dụ: Tại sao lại là Ác tính? Hãy tính gradient cho lớp Ác tính)
    score = output[0, target_class_index]
    score.backward()
    
    # 4. Lấy gradients và activations
    gradients = model.gradients # Lấy từ hook
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # 5. Nhân trọng số vào feature map
    activations = feature_map.detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # 6. Tạo heatmap (Trung bình cộng các kênh)
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # 7. ReLU (Chỉ lấy ảnh hưởng dương)
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    
    # 8. Chuẩn hóa về 0-1
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
        
    return heatmap

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_models():
    models_dict = {}
    if not os.path.exists(SEG_PATH): gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=False)
    if not os.path.exists(CLS_PATH): gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=False)
            
    try:
        seg_model = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1)
        seg_model.load_state_dict(torch.load(SEG_PATH, map_location=DEVICE))
        seg_model.eval()
        models_dict['seg'] = seg_model

        cls_model = BiradsNet_FineTune()
        cls_model.load_state_dict(torch.load(CLS_PATH, map_location=DEVICE))
        # Không eval() hoàn toàn vì cần gradient cho Grad-CAM, nhưng tắt dropout
        cls_model.train() 
        for module in cls_model.modules():
            if isinstance(module, nn.Dropout): module.eval()
            if isinstance(module, nn.BatchNorm2d): module.eval()
            
        models_dict['cls'] = cls_model
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None
    return models_dict

# --- 5. XỬ LÝ ẢNH ---
def process_image(image_pil):
    image_np = np.array(image_pil)
    transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    augmented = transform(image=image_np)
    img_tensor = augmented['image'].unsqueeze(0)
    return img_tensor, image_np

def overlay_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img * 0.6 # Độ trong suốt 40%
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

# --- 6. GIAO DIỆN ---
def main():
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117;} 
        .main-title {color: #ff4b4b; text-align: center; font-size: 3em;}
        .metric-box {border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🧬 TRUST-MED AI</h1>', unsafe_allow_html=True)
    st.info("Hệ thống chẩn đoán kép: Segmentation (Định vị) + Grad-CAM (Giải thích)")

    uploaded_file = st.file_uploader("Tải ảnh siêu âm...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        models = load_models()
        if models is None: return

        image_pil = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.warning("📷 1. Ảnh Gốc")
            st.image(image_pil, use_container_width=True)

        img_tensor, img_original_np = process_image(image_pil)
        
        # --- PHASE 1: SEGMENTATION ---
        with torch.no_grad():
            seg_out = models['seg'](img_tensor)
            pred_mask = (seg_out.sigmoid() > 0.5).float().cpu().numpy()[0,0]
        
        # Tính diện tích thực
        orig_h, orig_w = img_original_np.shape[:2]
        scale = (orig_w * orig_h) / (256 * 256)
        real_pixels = int(np.sum(pred_mask) * scale)

        # --- PHASE 2: CLASSIFICATION & GRAD-CAM ---
        # Chạy model phân loại (lấy output và feature map)
        output, _ = models['cls'](img_tensor)
        probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
        pred_idx = np.argmax(probs)
        real_birads = pred_idx + 2
        
        # Chỉ tạo Grad-CAM nếu có tổn thương
        if real_pixels > 50:
            # Tạo Grad-CAM cho class được dự đoán cao nhất
            heatmap = generate_gradcam(models['cls'], img_tensor, pred_idx)
            vis_gradcam = overlay_heatmap(heatmap, img_original_np)
        else:
            vis_gradcam = img_original_np # Không có u thì không cần heatmap

        # Hiển thị Grad-CAM
        with col2:
            st.info(f"🔥 2. AI Focus (Grad-CAM)")
            if real_pixels < 50:
                st.image(image_pil, caption="Không có vùng tập trung", use_container_width=True)
            else:
                st.image(vis_gradcam, caption="Vùng AI 'nhìn' để ra quyết định", use_container_width=True)

        # Hiển thị Kết quả
        with col3:
            st.success("📊 3. Kết quả Phân tích")
            
            # Logic hiển thị mới
            prob_benign = probs[0] + probs[1]
            prob_malignant = probs[2] + probs[3]
            
            if real_pixels < 50:
                st.markdown("### ✅ BÌNH THƯỜNG")
                st.caption("Không phát hiện tổn thương.")
            elif prob_benign > prob_malignant:
                st.markdown(f"### 🟢 LÀNH TÍNH ({prob_benign*100:.1f}%)")
                st.write(f"Khuyến nghị: **BI-RADS {real_birads}**")
                st.progress(float(prob_benign))
            else:
                st.markdown(f"### 🔴 ÁC TÍNH ({prob_malignant*100:.1f}%)")
                st.write(f"Khuyến nghị: **BI-RADS {real_birads}**")
                st.warning("Cần sinh thiết ngay.")
                st.progress(float(prob_malignant))

            st.write(f"- Diện tích u (ước tính): ~{real_pixels} px")
            st.bar_chart({"Lành": prob_benign, "Ác": prob_malignant})

if __name__ == "__main__":
    main()
