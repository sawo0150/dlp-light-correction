import os
import cv2
import numpy as np
import math
from pathlib import Path

# --- 설정 (Configuration) ---
IMG_SIZE = (160, 160)     # 전체 캔버스 크기
ROI_SIZE = 128            # 실제 패턴이 들어갈 영역
PAD = (IMG_SIZE[0] - ROI_SIZE) // 2  # 16px Padding
BASE_DIR = "data/benchmark_160"
BG_COLOR = 0              # Black
FG_COLOR = 255            # White

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_pair(category, filename, img):
    """
    Standard(원본)와 Inverse(반전) 이미지를 함께 저장합니다.
    단, Padding 영역(검은색)은 반전시키지 않고 유지합니다 (ROI만 반전).
    """
    save_dir = os.path.join(BASE_DIR, category)
    ensure_dir(save_dir)
    
    # 1. Standard Save
    path_std = os.path.join(save_dir, f"{filename}.png")
    cv2.imwrite(path_std, img)
    
    # 2. Inverse Save (Padding 보호)
    # 전체 반전 후 Padding 영역을 다시 0으로 강제
    inv_img = cv2.bitwise_not(img)
    mask = np.zeros_like(img)
    mask[PAD:PAD+ROI_SIZE, PAD:PAD+ROI_SIZE] = 255
    
    # 마스크 영역(ROI)만 반전된 이미지 가져오기 + 나머지는 0
    final_inv = cv2.bitwise_and(inv_img, mask)
    
    path_inv = os.path.join(save_dir, f"{filename}_inv.png")
    cv2.imwrite(path_inv, final_inv)
    
    print(f"Saved: {category}/{filename} (& _inv)")

def get_empty_canvas():
    return np.full(IMG_SIZE, BG_COLOR, dtype=np.uint8)

# ==========================================
# 1. Resolution Grid Test (Stripe -> Grid)
# ==========================================
def generate_grid_test():
    """
    격자(Grid) 패턴. 선 두께 2px 고정.
    Gap을 1px ~ 5px로 변화시키며 생성.
    중앙 90x90 영역 정도만 채움.
    """
    line_width = 2
    gaps = [1, 2, 3, 4, 5] # Gap Level
    
    # Grid를 그릴 중앙 박스 크기 (약 90~100px)
    grid_span = 96 
    start = (IMG_SIZE[0] - grid_span) // 2
    end = start + grid_span
    
    for gap in gaps:
        img = get_empty_canvas()
        step = line_width + gap
        
        # 수직선 (Vertical)
        for x in range(start, end, step):
            # 선 두께만큼 그리기
            cv2.rectangle(img, (x, start), (x + line_width - 1, end), FG_COLOR, -1)
            
        # 수평선 (Horizontal)
        for y in range(start, end, step):
            cv2.rectangle(img, (start, y), (end, y + line_width - 1), FG_COLOR, -1)
            
        save_pair("01_resolution", f"grid_gap_{gap}px", img)

# ==========================================
# 2. Geometry: Convex (Triangle)
# ==========================================
def draw_triangle(img, center, size, angle_deg):
    """
    중심점과 높이, 그리고 '꼭짓점 각도'를 받아 이등변 삼각형을 그림.
    angle_deg가 작을수록(10도) 매우 뾰족해짐.
    """
    cx, cy = center
    half_angle = math.radians(angle_deg / 2)
    
    # 높이(h)에 따른 밑변 절반 길이(w) 계산: tan(theta/2) = w / h
    height = size
    width_half = height * math.tan(half_angle)
    
    # 꼭짓점 (위쪽)
    pt_top = (cx, cy - height // 2)
    # 아래쪽 좌우
    pt_bl = (int(cx - width_half), int(cy + height // 2))
    pt_br = (int(cx + width_half), int(cy + height // 2))
    
    pts = np.array([pt_top, pt_br, pt_bl], np.int32)
    cv2.fillPoly(img, [pts], FG_COLOR)

def generate_convex_test():
    """
    각도가 다른(10, 20, 40도) 삼각형을 배치하여
    끝부분(Tip)이 얼마나 뭉개지는지(Erosion), 보정 후 복원되는지 확인.
    """
    # 1. 개별 각도 파일 생성
    angles = [10, 20, 40, 60]
    for ang in angles:
        img = get_empty_canvas()
        draw_triangle(img, (80, 80), 100, ang)
        save_pair("02_convex", f"angle_{ang:02d}", img)
        
    # 2. 한 화면에 모으기 (비교용)
    img_all = get_empty_canvas()
    # 좌(30), 중(80), 우(130) 배치 (ROI 16~144 내부에 들어옴)
    draw_triangle(img_all, (30, 80), 80, 10)  # Sharpest
    draw_triangle(img_all, (80, 80), 80, 20)
    draw_triangle(img_all, (130, 80), 80, 40)
    save_pair("02_convex", "angle_all_compare", img_all)

# ==========================================
# 3. Geometry: Concave (Notch)
# ==========================================
def draw_notched_rect(img, center, size, notch_angle_deg):
    """
    사각형의 한쪽 변에 쐐기(Notch) 모양으로 파인 형상.
    빛 번짐으로 인해 파인 부분이 채워지는지(Over-curing) 확인.
    """
    cx, cy = center
    half_size = size // 2
    
    # 기본 사각형 좌표
    tl = (cx - half_size, cy - half_size)
    tr = (cx + half_size, cy - half_size)
    br = (cx + half_size, cy + half_size)
    bl = (cx - half_size, cy + half_size)
    
    # Notch 깊이와 폭 계산
    notch_depth = int(size * 0.4)
    half_angle = math.radians(notch_angle_deg / 2)
    notch_width_half = notch_depth * math.tan(half_angle)
    
    # 오른쪽 변의 중앙에서 안쪽으로 파고듬
    notch_tip = (cx + half_size - notch_depth, cy) # 파인 꼭짓점
    notch_top = (cx + half_size, cy - int(notch_width_half))
    notch_bot = (cx + half_size, cy + int(notch_width_half))
    
    # Polygon 점 순서: 좌상 -> (오른쪽 위 -> 파인 점 -> 오른쪽 아래) -> 좌하
    pts = np.array([
        tl, 
        (cx + half_size, cy - half_size), # Top-Right Corner
        notch_top,
        notch_tip, # Notch Point (Concave Corner)
        notch_bot,
        br,
        bl
    ], np.int32)
    
    cv2.fillPoly(img, [pts], FG_COLOR)

def generate_concave_test():
    angles = [10, 20, 45, 90]
    for ang in angles:
        img = get_empty_canvas()
        draw_notched_rect(img, (80, 80), 100, ang)
        save_pair("03_concave", f"notch_{ang:02d}", img)

# ==========================================
# 4. Complex: Circle - (Big Triangle Cut) + (Small Triangle Insert)
# ==========================================
def generate_circle_tri_boolean():
    """
    Geometry(상호작용) 케이스를 단순화해서 04_complex로 통합.
    - 큰 원을 만든 뒤
    - '큰 삼각형'을 빼서(concave notch/void) 경계/코너 over-curing 취약점 유발
    - 그 자리/근처에 '작은 삼각형'을 넣어(thin feature) 미세 형상 재현성 테스트
    """
    img = get_empty_canvas()
    cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2

    # --- base circle mask ---
    circle = np.zeros_like(img)

    r_circle = 30
    cv2.circle(circle, (cx, cy), r_circle, 255, -1)

    # --- triangle mask (side ~ 30), slightly overlapping circle ---
    tri = np.zeros_like(img)
    side = 30.0
    h = int(round(side * math.sqrt(3) / 2.0))  # equilateral height (~26)

    # 배치: 삼각형이 원 오른쪽에 살짝 걸치도록 (겹침)
    # - 삼각형 중심을 원 중심에서 +dx로 이동 (dx가 클수록 겹침 감소)
    dx = 18
    tcx, tcy = cx + dx, cy

    # equilateral triangle (pointing left) to make a sharper interaction boundary
    # vertices:
    #   left tip, right-top, right-bottom
    pt_left  = (tcx - h, tcy)
    pt_rt    = (tcx + h // 2, tcy - int(round(side / 2.0)))
    pt_rb    = (tcx + h // 2, tcy + int(round(side / 2.0)))
    cv2.fillPoly(tri, [np.array([pt_left, pt_rt, pt_rb], np.int32)], 255)

    # --- dilated triangle ("puffed") to subtract from circle ---
    # dilation amount controls how much extra clearance you carve (simulating over-cure margin)
    dilate_px = 3
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tri_dil = cv2.dilate(tri, kernel, iterations=1)

    # Boolean: (circle - dilated_triangle) + (original_triangle)
    circle_cut = cv2.bitwise_and(circle, cv2.bitwise_not(tri_dil))
    final = cv2.bitwise_or(circle_cut, tri)

    # 캔버스에 적용
    img[final > 0] = FG_COLOR
    save_pair("04_complex", "circle_minus_diltri_plus_tri", img)
 

# ==========================================
# 5. Complex: Yin-Yang (음양)  -> 04_complex로 저장
# ==========================================
def generate_yin_yang():
    """
    음양(Yin-Yang) 패턴 생성.
    S자 곡선과 서로 다른 배경의 점(Dot) 테스트.
    """
    img = get_empty_canvas()
    cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
    r_big = 45      # 큰 원 반지름
    r_mid = r_big // 2
    r_small = 8     # 작은 점 반지름
    
    # 1. 큰 원 (흰색)
    cv2.circle(img, (cx, cy), r_big, FG_COLOR, -1)
    
    # 2. 오른쪽 반을 검은색으로 칠함 (반원 만들기)
    cv2.ellipse(img, (cx, cy), (r_big, r_big), 0, -90, 90, BG_COLOR, -1)
    
    # 3. 위쪽 중간 원 (검은색 배경 위 흰색 돌출 -> 사실 그냥 흰색 채우기)
    cv2.circle(img, (cx, cy - r_mid), r_mid, FG_COLOR, -1)
    
    # 4. 아래쪽 중간 원 (흰색 배경 위 검은색 파임)
    cv2.circle(img, (cx, cy + r_mid), r_mid, BG_COLOR, -1)
    
    # 5. 위쪽 작은 점 (흰색 배경 위 검은 점)
    cv2.circle(img, (cx, cy - r_mid), r_small, BG_COLOR, -1)
    
    # 6. 아래쪽 작은 점 (검은색 배경 위 흰 점)
    cv2.circle(img, (cx, cy + r_mid), r_small, FG_COLOR, -1)
    
    save_pair("04_complex", "yin_yang", img)

# ==========================================
# 6. Complex: Star Shape  -> 04_complex로 저장
# ==========================================
def generate_star():
    """
    Convex/Concave가 반복되는 별 모양.
    크기를 줄여서 ROI 내부에 안착.
    """
    img = get_empty_canvas()
    cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
    
    outer_radius = 45
    inner_radius = 20
    num_points = 6 # 육망성 느낌
    
    pts = []
    for i in range(num_points * 2):
        angle = math.pi * i / num_points - math.pi / 2 # 위쪽부터 시작
        r = outer_radius if i % 2 == 0 else inner_radius
        x = int(cx + math.cos(angle) * r)
        y = int(cy + math.sin(angle) * r)
        pts.append((x, y))
        
    cv2.fillPoly(img, [np.array(pts, np.int32)], FG_COLOR)
    save_pair("04_complex", "star_shape", img)

def generate_complex_suite():
    """
    기존 02_geometry + 03_complex를 통합한 04_complex 세트.
    - circle_cut_insert_tri (boolean interaction)
    - yin_yang
    - star_shape
    """
    generate_circle_tri_boolean()
    generate_yin_yang()
    generate_star()

def main():
    print(f"Generating Benchmark (160x160 with 16px Padding)...")
    
    if os.path.exists(BASE_DIR):
        print(f"Warning: '{BASE_DIR}' already exists. Files may be overwritten.")
    
    generate_grid_test()              # 01_resolution (Grid Gap 1~5px)
    generate_convex_test()            # 02_convex (Triangle Angles)
    generate_concave_test()           # 03_concave (Notched Rect)
    generate_complex_suite()          # 04_complex (merged: geometry + complex)
    
    print("All Done! Check 'data/benchmark_160'")

if __name__ == "__main__":
    main()