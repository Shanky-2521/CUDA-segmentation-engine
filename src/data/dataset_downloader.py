import os
import zipfile
import requests
from tqdm import tqdm
import kaggle
from pathlib import Path
from config import DATA_DIR

class CityscapesDownloader:
    def __init__(self):
        self.dataset_name = "shuvoalok/cityscapes"
        self.data_dir = Path(DATA_DIR)
        self.kaggle_api = kaggle.api
        
    def download_dataset(self):
        """
        Download Cityscapes dataset from Kaggle
        """
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        print("Downloading Cityscapes dataset...")
        self.kaggle_api.dataset_download_files(
            self.dataset_name,
            path=str(self.data_dir),
            unzip=True
        )
        print("Dataset downloaded successfully!")
        
    def organize_data(self):
        """
        Organize the downloaded data into the correct Cityscapes structure
        """
        print("Organizing dataset...")
        
        # Create necessary directories
        (self.data_dir / 'leftImg8bit' / 'train').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'leftImg8bit' / 'val').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'gtFine' / 'train').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'gtFine' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Get list of all downloaded files
        for split in ['train', 'val']:
            # Handle images
            image_dir = self.data_dir / split / 'img'
            label_dir = self.data_dir / split / 'label'
            
            # Create city directories
            for city_name in ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf',
                             'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach',
                             'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']:
                (self.data_dir / 'leftImg8bit' / split / city_name).mkdir(parents=True, exist_ok=True)
                (self.data_dir / 'gtFine' / split / city_name).mkdir(parents=True, exist_ok=True)
            
            # Process images
            for image_file in image_dir.glob('*.png'):
                try:
                    # Extract city name from filename
                    city_name = image_file.stem.split('_')[0]
                    if city_name in ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf',
                                    'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach',
                                    'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']:
                        target_dir = self.data_dir / 'leftImg8bit' / split / city_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        image_file.rename(target_dir / image_file.name)
                    else:
                        print(f"Skipping invalid city name: {city_name}")
                except Exception as e:
                    print(f"Error moving image {image_file}: {str(e)}")
            
            # Process labels
            for label_file in label_dir.glob('*.png'):
                try:
                    # Extract city name from filename
                    city_name = label_file.stem.split('_')[0]
                    if city_name in ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf',
                                    'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach',
                                    'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']:
                        target_dir = self.data_dir / 'gtFine' / split / city_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        label_file.rename(target_dir / label_file.name)
                    else:
                        print(f"Skipping invalid city name: {city_name}")
                except Exception as e:
                    print(f"Error moving label {label_file}: {str(e)}")
        
        print("\nDataset organization complete!")
        
        # Verify dataset structure
        self.verify_dataset()
        
        print("Dataset organization complete!")
        
    def verify_dataset(self):
        """
        Verify dataset structure and contents
        """
        print("\nVerifying dataset structure...")
        
        # Check splits
        splits = ['train', 'val']
        for split in splits:
            # Check images
            image_dir = self.data_dir / 'leftImg8bit' / split
            num_images = len(list(image_dir.glob('*/*.png')))
            
            # Check labels
            label_dir = self.data_dir / 'gtFine' / split
            num_labels = len(list(label_dir.glob('*/*.png')))
            
            print(f"\nSplit: {split}")
            print(f"Number of images: {num_images}")
            print(f"Number of labels: {num_labels}")
            
            # Check if numbers match
            if num_images != num_labels:
                print("Warning: Number of images and labels don't match!")
        
        print("\nDataset verification complete!")

def main():
    # Initialize downloader
    downloader = CityscapesDownloader()
    
    # Download and organize dataset
    downloader.download_dataset()
    downloader.organize_data()
    downloader.verify_dataset()

if __name__ == '__main__':
    main()
