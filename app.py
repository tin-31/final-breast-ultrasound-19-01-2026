import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
import os
import math

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="TRUST-MED: AI Siêu âm Vú", page_icon="🎗️", layout="wide")

SEG_FILE_ID = '1WYMlDSjXCnPE21C2jy6jRk7NlZGEnaCm' 
CLS_FILE_ID = '1P2nuQ9HbJliaItRP-F9oeVmcDhYZ4Ju8'
SEG_PATH = 'TRUST_MED_FINAL_ROBUST.pth'
CLS_PATH = 'TRUST_MED_BIRADS_EXPERT_FINETUNED.pth'
DEVICE = 'cpu'

# --- 2. KIẾN TRÚC MODEL & GRAD-CAM HOOK ---
class BiradsNet_FineTune(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        features_list = self.encoder(x)
        x = features_list[-1]
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = self.head(x)
        return x, features_list[-1]

# --- 3. CÁC HÀM XỬ LÝ CHUYÊN SÂU ---

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
        cls_model.train() # Hack để lấy gradient cho Grad-CAM
        for module in cls_model.modules():
            if isinstance(module, nn.Dropout): module.eval()
            if isinstance(module, nn.BatchNorm2d): module.eval()
        models_dict['cls'] = cls_model
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None
    return models_dict

def process_image(image_pil):
    image_np = np.array(image_pil)
    transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    augmented = transform(image=image_np)
    img_tensor = augmented['image'].unsqueeze(0)
    return img_tensor, image_np

def generate_gradcam(model, img_tensor, target_class_index):
    model.eval()
    img_tensor.requires_grad = True
    output, feature_map = model(img_tensor)
    model.zero_grad()
    score = output[0, target_class_index]
    score.backward()
    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = feature_map.detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img * 0.6
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def calculate_geometry(mask_256, original_shape):
    """Tính toán Diện tích và Chu vi dựa trên Mask"""
    # Resize mask về kích thước gốc
    mask_resized = cv2.resize(mask_256, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_uint8 = mask_resized.astype(np.uint8)
    
    # Tìm đường viền (Contours)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, 0, mask_uint8
        
    # Lấy đường viền lớn nhất (khối u chính)
    cnt = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Tính độ tròn (Roundness): 4*pi*A / P^2. (1.0 là tròn vo, < 0.5 là méo)
    roundness = 0
    if perimeter > 0:
        roundness = (4 * math.pi * area) / (perimeter * perimeter)
        
    return area, perimeter, roundness, mask_uint8

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117;} 
        .main-title {color: #ff4b4b; text-align: center; font-size: 3em; font-weight: bold;}
        .sub-header {color: #ffffff; text-align: center;}
        .metric-card {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #333;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🧬 TRUST-MED AI</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải ảnh siêu âm...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        models = load_models()
        if models is None: return

        image_pil = Image.open(uploaded_file).convert('RGB')
        img_tensor, img_original_np = process_image(image_pil)
        
        # --- PHASE 1: SEGMENTATION & GEOMETRY ---
        with torch.no_grad():
            seg_out = models['seg'](img_tensor)
            pred_mask = (seg_out.sigmoid() > 0.5).float().cpu().numpy()[0,0]
            
        area, perimeter, roundness, mask_full_size = calculate_geometry(pred_mask, img_original_np.shape)

        # --- PHASE 2: CLASSIFICATION ---
        output, _ = models['cls'](img_tensor)
        probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
        bi_idx = np.argmax(probs)
        real_birads = bi_idx + 2
        prob_benign = probs[0] + probs[1]
        prob_malignant = probs[2] + probs[3]

        # --- GIAO DIỆN 3 CỘT ---
        col1, col2, col3 = st.columns([1, 1.2, 1])
        
        # CỘT 1: ẢNH GỐC
        with col1:
            st.warning("📷 1. Ảnh Gốc")
            st.image(image_pil, use_container_width=True)
            
            # Hiển thị thông số hình học
            st.markdown("### 📏 Thông số đo đạc")
            if area > 50:
                st.markdown(f"""
                <div class="metric-card">
                    <p><b>Diện tích:</b> {int(area)} px²</p>
                    <p><b>Chu vi:</b> {int(perimeter)} px</p>
                    <p><b>Độ tròn:</b> {roundness:.2f} { "(Tròn)" if roundness > 0.6 else "(Méo/Gai)" }</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Chưa đo được (Không có u)")

        # CỘT 2: PHÂN TÍCH HÌNH ẢNH (TAB)
        with col2:
            tab1, tab2 = st.tabs(["🔴 Phân đoạn (Segmentation)", "🔥 Giải thích (XAI)"])
            
            with tab1:
                # Vẽ Segmentation chồng lên ảnh
                if area > 50:
                    overlay = img_original_np.copy()
                    # Vẽ màu đỏ
                    overlay[mask_full_size == 1] = [255, 0, 0]
                    # Vẽ viền vàng cho rõ
                    contours, _ = cv2.findContours(mask_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
                    
                    final_seg = cv2.addWeighted(img_original_np, 0.7, overlay, 0.3, 0)
                    st.image(final_seg, caption="AI khoanh vùng tổn thương", use_container_width=True)
                else:
                    st.image(image_pil, caption="Không phát hiện khối u")
            
            with tab2:
                # Grad-CAM
                if area > 50:
                    heatmap = generate_gradcam(models['cls'], img_tensor, bi_idx)
                    vis_gradcam = overlay_heatmap(heatmap, img_original_np)
                    st.image(vis_gradcam, caption="Vùng AI tập trung (Đỏ = Quan trọng)", use_container_width=True)
                    st.caption("ℹ️ *Grad-CAM hiển thị vùng đặc trưng (bờ, gai, âm) mà model dùng để chẩn đoán.*")
                else:
                    st.info("Không có khối u để phân tích XAI.")

        # CỘT 3: KẾT LUẬN & BI-RADS
        with col3:
            st.success("📊 3. Kết luận")
            
            if area < 50:
                st.markdown("# ✅ BÌNH THƯỜNG")
                st.info("Không phát hiện tổn thương khu trú.")
                st.metric("BI-RADS", "1")
            
            elif prob_benign > prob_malignant:
                st.markdown(f"# 🟢 LÀNH TÍNH")
                st.progress(float(prob_benign))
                st.caption(f"Độ tin cậy: {prob_benign*100:.1f}%")
                
                st.markdown("---")
                st.metric("Phân loại BI-RADS", f"BI-RADS {real_birads}")
                st.info("💡 **Khuyến nghị:** Theo dõi định kỳ (Follow-up).")
                
            else:
                st.markdown(f"# 🔴 ÁC TÍNH")
                st.progress(float(prob_malignant))
                st.caption(f"Độ tin cậy: {prob_malignant*100:.1f}%")
                
                st.markdown("---")
                st.metric("Phân loại BI-RADS", f"BI-RADS {real_birads}")
                st.error("⚡ **Khuyến nghị:** Cần SINH THIẾT (Biopsy) ngay.")

            # Biểu đồ chi tiết
            with st.expander("Xem chi tiết xác suất"):
                st.bar_chart({"Lành (2/3)": prob_benign, "Ác (4/5)": prob_malignant})

if __name__ == "__main__":
    main()
