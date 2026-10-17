import os
import requests
import sys
from tqdm import tqdm

BASE_URL = "https://rangeland.ntsg.umt.edu/data/naip-chm/inference-resources"

def download_file(url, dest_path):
    """
    Download a file from a URL to a destination path with a progress bar.

    Writes to a temporary .part file and renames it on success, so a failed or
    interrupted download does not leave a partial file that later runs would skip.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    desc = os.path.basename(dest_path)
    
    tmp_path = dest_path + ".part"
    try:
        with open(tmp_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        os.replace(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    # Base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Conditioning data configuration
    cond_data_dir = os.path.join(data_dir, 'conditioning_data')
    cond_urls = {
        "climate_pca.tif": f"{BASE_URL}/conditioning-data/climate_pca.tif",
        "ecoregion.tif": f"{BASE_URL}/conditioning-data/ecoregion.tif",
        "elevation.tif": f"{BASE_URL}/conditioning-data/elevation.tif",
        "nlcd.tif": f"{BASE_URL}/conditioning-data/nlcd.tif",
        "soil_pca.tif": f"{BASE_URL}/conditioning-data/soil_pca.tif"
    }
    
    # Create directory if it doesn't exist
    os.makedirs(cond_data_dir, exist_ok=True)
    
    print(f"Downloading conditioning data to {cond_data_dir}...")
    
    for filename, url in cond_urls.items():
        dest_path = os.path.join(cond_data_dir, filename)
        if os.path.exists(dest_path):
            print(f"Skipping {filename} (already exists)")
            continue
            
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    # NAIP DOQQ configuration
    naip_url = f"{BASE_URL}/naip-doqq-example/m_3812259_nw_10_060_20220519.tif"
    naip_filename = "m_3812259_nw_10_060_20220519.tif"
    naip_dir = os.path.join(data_dir, 'naip_doqqs')
    
    print("\nOptional: Sample NAIP DOQQ Download")
    response = input("Do you want to download the sample NAIP DOQQ to test the inference pipeline? (y/N): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        os.makedirs(naip_dir, exist_ok=True)
        dest_path = os.path.join(naip_dir, naip_filename)
        
        if os.path.exists(dest_path):
            print(f"Skipping {naip_filename} (already exists)")
        else:
            print(f"Downloading sample NAIP DOQQ to {naip_dir}...")
            try:
                download_file(naip_url, dest_path)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading NAIP DOQQ: {e}")
    else:
        print("Skipping NAIP DOQQ download.")

if __name__ == "__main__":
    main()
