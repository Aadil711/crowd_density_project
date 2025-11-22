"""
Run CSRNet inference on extracted frames to count crowds (GPU/batch optimized).
Usage: python scripts/count_crowd.py anonymized_frames/street1 models/preds/street1 [weights_path]
"""
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from csrnet_model import load_model
from tqdm import tqdm

def preprocess_image_for_batch(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.transpose((2, 0, 1))  # HWC to CHW
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std
    return img.astype(np.float32)

def save_density_map(density_map, output_path):
    dmap = density_map.squeeze().cpu().numpy()
    dmap = (dmap - dmap.min()) / (dmap.max() - dmap.min() + 1e-5)
    dmap = (dmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path), heatmap)

def process_frames(frames_dir, output_dir, weights_path, save_heatmaps=True, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_heatmaps:
        heatmaps_dir = output_dir / "heatmaps"
        heatmaps_dir.mkdir(exist_ok=True)
    else:
        heatmaps_dir = None

    print(f"Using device: {device}")

    # Load model ONCE on GPU or CPU
    print("Loading CSRNet model...")
    model = load_model(weights_path).to(device)
    model.eval()
    
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    total_frames = len(frame_files)
    print(f"Found {total_frames} frames to process (batch_size={batch_size})")

    results = []
    for start_idx in tqdm(range(0, total_frames, batch_size), desc="Batch Inference"):
        batch_files = frame_files[start_idx:start_idx+batch_size]
        batch_images = [preprocess_image_for_batch(fp) for fp in batch_files]
        batch_tensor = torch.from_numpy(np.stack(batch_images)).float().to(device)

        with torch.no_grad():
            density_maps = model(batch_tensor)
            counts = density_maps.view(density_maps.size(0), -1).sum(dim=1).cpu().numpy()

        for i, (fp, count, density_map) in enumerate(zip(batch_files, counts, density_maps)):
            idx = start_idx + i
            if save_heatmaps:
                heatmap_path = heatmaps_dir / f"{fp.stem}_heatmap.jpg"
                save_density_map(density_map, heatmap_path)
            result = {
                "frame": str(fp),
                "frame_name": fp.name,
                "count": round(float(count), 2),
                "frame_index": int(idx)
            }
            results.append(result)

    print(f"\n✅ Processed {len(results)} frames")

    output_json = output_dir / "counts.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {output_json}")

    counts = [r['count'] for r in results]
    print(f"\n📊 Summary:")
    print(f"   Average count: {np.mean(counts):.2f}")
    print(f"   Min count: {np.min(counts):.2f}")
    print(f"   Max count: {np.max(counts):.2f}")

    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/count_crowd.py <frames_dir> <output_dir> [weights_path]")
        print("Example: python scripts/count_crowd.py anonymized_frames/street1 models/preds/street1")
        sys.exit(1)

    frames_dir = sys.argv[1]
    output_dir = sys.argv[2]
    weights_path = sys.argv[3] if len(sys.argv) > 3 else None
    # You can change batch_size for your card: 8 or 16 works for most RTX
    process_frames(frames_dir, output_dir, weights_path, save_heatmaps=True, batch_size=8)
