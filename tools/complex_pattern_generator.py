import os
import cv2
import numpy as np
import glob
from pathlib import Path

# --- 설정 (Configuration) ---
IMG_SIZE = (160, 160)     # 전체 캔버스 크기
ROI_SIZE = 128            # 실제 패턴이 들어갈 영역 (1280 -> 128로 축소)
PAD = (IMG_SIZE[0] - ROI_SIZE) // 2  # 16px Padding
BASE_DIR = "data/benchmark_160"

# 원본 이미지가 있는 경로 (사용자 환경에 맞게 수정하세요)
SOURCE_DIR = "/home/wosasa/Downloads/complex_pattern" 

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_pair(category, filename, img):
    """
    기존 코드와 동일: Standard 및 Inverse 저장 (Padding 영역 보호)
    """
    save_dir = os.path.join(BASE_DIR, category)
    ensure_dir(save_dir)
    
    # 1. Standard Save
    path_std = os.path.join(save_dir, f"{filename}.png")
    cv2.imwrite(path_std, img)
    
    # 2. Inverse Save (Padding 보호)
    inv_img = cv2.bitwise_not(img)
    mask = np.zeros_like(img)
    mask[PAD:PAD+ROI_SIZE, PAD:PAD+ROI_SIZE] = 255
    
    final_inv = cv2.bitwise_and(inv_img, mask)
    
    path_inv = os.path.join(save_dir, f"{filename}_inv.png")
    cv2.imwrite(path_inv, final_inv)
    print(f"Saved: {category}/{filename} (& _inv)")

def process_real_patterns():
    """
    1280x1280 이미지를 불러와 128x128로 리사이징하고 
    160x160 캔버스 중앙에 배치합니다.
    """
    # 1. 이미지 파일 목록 가져오기 (png, jpg 등)
    # glob을 사용하여 폴더 내의 모든 png 파일을 찾습니다.
    search_path = os.path.join(SOURCE_DIR, "*.png")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Error: 경로에 이미지가 없습니다 -> {search_path}")
        return

    print(f"Found {len(files)} images in {SOURCE_DIR}")

    for file_path in files:
        # 파일명 추출 (확장자 제거)
        filename = Path(file_path).stem
        
        # 2. 이미지 로드 (Grayscale)
        src_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if src_img is None:
            print(f"Skipped (Load Fail): {filename}")
            continue
            
        # 3. Downsampling (1280 -> 128)
        # INTER_AREA는 이미지를 축소할 때 모아레(Moiré) 패턴이나 정보 손실을 줄이는 데 가장 좋습니다.
        resized_img = cv2.resize(src_img, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
        
        # 4. Re-Binarization (중요)
        # 리사이징 과정에서 경계면이 흐릿해질 수 있으므로(Gray값 발생), 다시 0과 255로 강제합니다.
        # 127보다 크면 255(흰색), 아니면 0(검은색)
        _, bin_img = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)
        
        # 5. Canvas Embedding
        canvas = np.full(IMG_SIZE, 0, dtype=np.uint8) # 검은색 배경
        
        # 중앙 배치 (Padding 영역 제외한 안쪽에 넣기)
        canvas[PAD:PAD+ROI_SIZE, PAD:PAD+ROI_SIZE] = bin_img
        
        # 6. 저장 (05_real_pattern 폴더에 저장)
        save_pair("05_real_pattern", filename, canvas)

if __name__ == "__main__":
    process_real_patterns()