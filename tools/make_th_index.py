# tools/make_index.py
import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ---------------------------------------------------------
# 설정: 데이터셋 경로를 본인 환경에 맞게 수정하세요
# TRAINPACK_ROOT = "/home/swpants05/Desktop/trainpacks/TrainPack_allinone_COPY_1280"
TRAINPACK_ROOT = "/home/wosasa/Desktop/25-1_UROP/MiniPack_allinone_COPY_1280"
# 스캔할 하위 폴더들 (여기에 있는 모든 이미지를 긁어옵니다)
TARGET_DIRS = [
    "thr/binary/random",
    "thr/gray/random",
    # 필요한 경우 다른 폴더 추가
]
OUTPUT_FILENAME = "thr_file_index.json"
# ---------------------------------------------------------

def make_index():
    root = Path(TRAINPACK_ROOT)
    index = defaultdict(list)
    
    print(f"Scanning directories in {root}...")

    total_files = 0
    
    for rel_dir in TARGET_DIRS:
        target_dir = root / rel_dir
        if not target_dir.exists():
            print(f"Warning: Directory not found, skipping: {target_dir}")
            continue
            
        print(f"Scanning {rel_dir}...")
        
        # os.scandir이 os.listdir보다 속도가 빠릅니다.
        with os.scandir(target_dir) as entries:
            for entry in tqdm(entries, desc=f"Indexing {rel_dir}"):
                if not entry.name.endswith(".png"):
                    continue
                
                # 파일명 파싱 로직
                # 예: B5_..._mask__R01__80.png
                # 핵심: "__R"을 기준으로 앞부분이 sample_key임
                name = entry.name
                if "__R" in name:
                    sample_key = name.split("__R")[0]
                    
                    # 나중에 경로 결합을 위해 '폴더/파일명' 형태로 저장
                    # (전체 절대경로로 저장하면 용량이 커지므로 상대경로 권장)
                    full_rel_path = f"{rel_dir}/{name}"
                    index[sample_key].append(full_rel_path)
                    total_files += 1

    # 정렬 (R00, R01 순서 보장을 위해)
    print("Sorting file lists...")
    for key in index:
        index[key].sort()

    output_path = root / OUTPUT_FILENAME
    print(f"Saving index to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=None)  # indent 없어야 용량 작음

    print(f"Done! Indexed {total_files} files for {len(index)} keys.")

if __name__ == "__main__":
    make_index()