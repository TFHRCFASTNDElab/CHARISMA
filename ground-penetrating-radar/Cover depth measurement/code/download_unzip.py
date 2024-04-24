import os
from tqdm import tqdm
import urllib.request
import zipfile

def download_data_from_links(file_path, download_dir):
    try:
        # Check if the download directory exists, if not create it
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()[1:]  # Read all lines except the first one
            for line in lines:
                # Strip leading and trailing whitespace
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Extracting the download link from each line
                download_link = line
                # Extracting the zone number from the URL
                zone_number = download_link.split('=')[-1].split('%20')[-1].split('.')[0]
                # Constructing the filename with "GPR Zone"
                file_name = f"GPR Zone {zone_number}.zip"
                # Downloading the file to the specified directory
                with tqdm(unit='B', unit_scale=True, desc=f"Downloading {file_name}") as pbar:
                    urllib.request.urlretrieve(download_link, os.path.join(download_dir, file_name), reporthook=lambda blocknum, blocksize, totalsize: pbar.update(blocksize))
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def unzip_all(directory):
    """
    Extracts all contents of zip files in the specified directory.
    
    Parameters:
    - directory (str): The directory containing the zip files.
    """
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            # Get the full path of the zip file
            filepath = os.path.join(directory, filename)
            
            # Open the zip file
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                # Use tqdm to show progress
                # Set total size of the archive for tqdm to accurately show progress
                total_size = sum((file.file_size for file in zip_ref.infolist()))
                # Extract all contents to the current directory
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Extracting {filename}') as pbar:
                    for file in zip_ref.infolist():
                        zip_ref.extract(file, directory)
                        pbar.update(file.file_size)

            print(f"Extracted contents from {filename}")