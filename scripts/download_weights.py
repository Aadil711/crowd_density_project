
import urllib.request
import ssl
from pathlib import Path

def download_with_progress(url, output_path):
    """Download file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading from:\n{url}")
    print(f"Saving to: {output_path}\n")
    
    # Create SSL context to handle HTTPS
    context = ssl._create_unverified_context()
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100.0 / total_size if total_size > 0 else 0)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook, context=context)
        print(f"\n✅ Download complete: {output_path}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    # Using Google Drive link for reliable CSRNet weights
    # This is from a well-known CSRNet implementation
    
    print("=" * 60)
    print("CSRNet Pretrained Weights Downloader")
    print("=" * 60)
    
    # Try HuggingFace model hub (more reliable than GitHub)
    url = "https://huggingface.co/rootstrap-org/crowd-counting/resolve/main/pytorch_model.bin"
    output_path = "models/csrnet_shanghaitech.pth"
    
    success = download_with_progress(url, output_path)
    
    if not success:
        print("\n" + "=" * 60)
        print("ALTERNATIVE: Manual Download Instructions")
        print("=" * 60)
        print("\n1. Open your browser and go to:")
        print("   https://huggingface.co/rootstrap-org/crowd-counting")
        print("\n2. Click on 'Files and versions' tab")
        print("\n3. Download 'pytorch_model.bin'")
        print("\n4. Save it as: models/csrnet_shanghaitech.pth")
        print("\n5. Then run:")
        print("   python scripts/count_crowd.py anonymized_frames/street1 models/preds/street1 models/csrnet_shanghaitech.pth")
