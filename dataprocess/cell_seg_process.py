import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 參數
input_root  = Path("/orange/ruogu.fang/tienyuchang/CellData/OCT")
output_root = Path("/orange/ruogu.fang/tienyuchang/CellData/OCT_tri")
output_root.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpeg"}
COPY_TIMES = 3
MAX_WORKERS = min(32, (os.cpu_count() or 8) * 2)  # I/O-bound：多一點 thread

# 收集影像
image_paths = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
print(f"共找到 {len(image_paths)} 張影像。")

def copy_triplicate(src: Path):
    """為單張 src 建資料夾並複製三份；回傳 (src, ok, err_msg)"""
    try:
        rel_path = src.relative_to(input_root)
        out_dir = output_root / rel_path.parent / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, COPY_TIMES + 1):
            dst = out_dir / f"{src.stem}_{i}{src.suffix}"
            # 若不想覆蓋，保留這行；想覆蓋就拿掉
            if dst.exists():
                continue
            shutil.copy2(src, dst)

        return (src, True, "")
    except Exception as e:
        return (src, False, str(e))

# 併發執行
errors = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(copy_triplicate, p) for p in image_paths]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="複製中"):
        src, ok, msg = fut.result()
        if not ok:
            errors.append((str(src), msg))

print("✅ Done! 所有圖片已嘗試複製到新 output_root 並建立三份。")
if errors:
    print(f"⚠️ 有 {len(errors)} 個檔案失敗，例如：")
    for s, m in errors[:10]:
        print(" -", s, "=>", m)


