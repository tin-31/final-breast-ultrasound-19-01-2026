import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
import os
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="TRUST-MED: AI Chẩn Đoán Ung Thư Vú",
    page_icon="🎗️",
    layout="wide"
)

# ID Google Drive (Đã cập nhật từ link bạn gửi)
SEG_FILE_ID = '1WYMlDSjXCnPE21C2jy6jRk7NlZGEnaCm' 
CLS_FILE_ID = '1P2nuQ9HbJliaItRP-F9oeVmcDhYZ4Ju8'

SEG_PATH = 'TRUST_MED_FINAL_ROBUST.pth'
CLS_PATH = 'TRUST_MED_BIRADS_EXPERT_FINETUNED.pth'

DEVICE = 'cpu' # Cloud dùng CPU để tiết kiệm chi phí

# --- 2. ĐỊNH NGHĨA KIẾN TRÚC MODEL (BẮT BUỘC ĐỂ LOAD ĐƯỢC) ---

class BiradsNet_FineTune(nn.Module):
    """Kiến trúc mạng nơ-ron chẩn đoán BI-RADS"""
    def __init__(self):
        super().__init__()
        # Khởi tạo Encoder giống hệt lúc train
        self.encoder = smp.encoders.get_encoder(
            name="efficientnet-b0", in_channels=3, depth=5, weights=None
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        out_channels = self.encoder.out_channels[-1]
        
        # Bộ não phân loại
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4) # 4 lớp: BI-RADS 2,3,4,5
        )
        
    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        x = self.avgpool(x)
        x = self.head(x)
        return x

# --- 3. HÀM TẢI VÀ LOAD MODEL TỰ ĐỘNG ---
@st.cache_resource
def load_models():
    models_dict = {}
    
    # A. Tải & Load Model Segmentation (U-Net)
    if not os.path.exists(SEG_PATH):
        with st.spinner('Đang tải dữ liệu Phân đoạn từ Cloud... (Lần đầu sẽ hơi lâu)'):
            url = f'https://drive.google.com/uc?id={SEG_FILE_ID}'
            gdown.download(url, SEG_PATH, quiet=False)
            
    try:
        # Khởi tạo kiến trúc rỗng trước
        seg_model = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1)
        # Load trọng số vào
        seg_model.load_state_dict(torch.load(SEG_PATH, map_location=DEVICE))
        seg_model.eval()
        models_dict['seg'] = seg_model
    except Exception as e:
        st.error(f"Lỗi load model Segmentation: {e}")
        return None

    # B. Tải & Load Model Classification (BI-RADS)
    if not os.path.exists(CLS_PATH):
        with st.spinner('Đang tải dữ liệu Chẩn đoán từ Cloud...'):
            url = f'https://drive.google.com/uc?id={CLS_FILE_ID}'
            gdown.download(url, CLS_PATH, quiet=False)
            
    try:
        cls_model = BiradsNet_FineTune()
        cls_model.load_state_dict(torch.load(CLS_PATH, map_location=DEVICE))
        cls_model.eval()
        models_dict['cls'] = cls_model
    except Exception as e:
        st.error(f"Lỗi load model BI-RADS: {e}")
        return None
        
    return models_dict

# --- 4. HÀM XỬ LÝ ẢNH ---
def process_image(image_pil):
    image_np = np.array(image_pil)
    # Resize về 256x256 chuẩn input của model
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=image_np)
    img_tensor = augmented['image'].unsqueeze(0) # Thêm batch dim [1, 3, 256, 256]
    return img_tensor, image_np

# --- 5. GIAO DIỆN NGƯỜI DÙNG (UI) ---
def main():
    # CSS tùy chỉnh cho đẹp
    st.markdown("""
        <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #d63384;}
        .stButton>button {width: 100%; background-color: #d63384; color: white;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎗️ TRUST-MED: AI Chẩn Đoán Ung Thư Vú")
    st.markdown("**Hệ thống hỗ trợ ra quyết định lâm sàng (CDSS) dựa trên Deep Learning**")
    st.info("Phiên bản Demo: Tích hợp Segmentation (U-Net) & Classification (EfficientNet Fine-tuned).")
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Tải ảnh siêu âm (JPG, PNG)...", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        models = load_models()
        if models is None: return

        image_pil = Image.open(uploaded_file).convert('RGB')
        
        # Chia cột giao diện
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Ảnh Siêu âm gốc")
            st.image(image_pil, use_container_width=True)

        # --- BẮT ĐẦU PHÂN TÍCH ---
        with st.spinner('AI đang quét tổn thương...'):
            img_tensor, img_original_np = process_image(image_pil)
            
            # Bước 1: Segmentation (Tìm u)
            with torch.no_grad():
                seg_out = models['seg'](img_tensor)
                # Ngưỡng 0.5 để tạo mask nhị phân
                pred_mask = (seg_out.sigmoid() > 0.5).float().cpu().numpy()[0,0]
            
            tumor_area = np.sum(pred_mask)

        # Hiển thị kết quả Segmentation
        with col2:
            st.subheader("2. Vùng nghi ngờ (ROI)")
            if tumor_area < 50:
                st.image(image_pil, caption="Không phát hiện vùng u rõ ràng", use_container_width=True)
            else:
                # Vẽ mask đỏ lên ảnh
                mask_resized = cv2.resize(pred_mask, (img_original_np.shape[1], img_original_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay = img_original_np.copy()
                overlay[mask_resized == 1] = [255, 0, 0] # Màu đỏ
                result_img = cv2.addWeighted(img_original_np, 0.7, overlay, 0.3, 0)
                st.image(result_img, caption=f"Diện tích u: {int(tumor_area)} pixels", use_container_width=True)

        # Bước 2: Classification (Chẩn đoán BI-RADS)
        with col3:
            st.subheader("3. Kết luận & Khuyến nghị")
            
            # LOGIC SÀNG LỌC THÔNG MINH
            if tumor_area < 50:
                st.success("✅ **KẾT QUẢ: BI-RADS 1 (BÌNH THƯỜNG)**")
                st.write("Không phát hiện tổn thương khu trú.")
                st.info("💡 Khuyến nghị: Sàng lọc định kỳ hàng năm.")
            else:
                # Chỉ chạy model phân loại khi có u
                with torch.no_grad():
                    cls_out = models['cls'](img_tensor)
                    probs = torch.softmax(cls_out, dim=1).cpu().numpy()[0]
                    # Map: 0->2, 1->3, 2->4, 3->5
                    pred_idx = np.argmax(probs)
                    real_birads = pred_idx + 2 
                
                # Hiển thị kết quả chi tiết
                if real_birads == 2:
                    st.success(f"🟢 **BI-RADS {real_birads}: LÀNH TÍNH**")
                    st.write("Tổn thương có đặc điểm lành tính (nang, nhân xơ).")
                    st.info("💡 Khuyến nghị: Theo dõi định kỳ, không can thiệp.")
                    
                elif real_birads == 3:
                    st.warning(f"🟡 **BI-RADS {real_birads}: KHẢ NĂNG LÀNH TÍNH**")
                    st.write("Tỷ lệ ác tính thấp (<2%).")
                    st.info("💡 Khuyến nghị: Theo dõi ngắn hạn (6 tháng).")
                    
                elif real_birads == 4:
                    st.error(f"🟠 **BI-RADS {real_birads}: NGHI NGỜ ÁC TÍNH**")
                    st.write("Tổn thương có dấu hiệu nghi ngờ.")
                    st.error("⚡ **Khuyến nghị: Cần SINH THIẾT (Biopsy) để xác chẩn.**")
                    
                elif real_birads == 5:
                    st.error(f"🔴 **BI-RADS {real_birads}: RẤT NGHI NGỜ ÁC TÍNH**")
                    st.write("Hình thái điển hình của ung thư (>95%).")
                    st.error("⚡ **Khuyến nghị: SINH THIẾT NGAY và hội chẩn Ung bướu.**")
                
                # Biểu đồ xác suất
                st.markdown("---")
                st.caption("Phân phối xác suất AI:")
                st.bar_chart({
                    "BI-RADS 2": probs[0],
                    "BI-RADS 3": probs[1],
                    "BI-RADS 4": probs[2],
                    "BI-RADS 5": probs[3]
                })

if __name__ == "__main__":
    main()
