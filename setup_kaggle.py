import os
import json
from pathlib import Path

def setup_kaggle():
    """
    Set up Kaggle API credentials
    """
    # Create .kaggle directory if it doesn't exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json with credentials
    kaggle_json = {
        "username": "YOUR_KAGGLE_USERNAME",
        "key": "YOUR_KAGGLE_API_KEY"
    }
    
    # Save credentials
    with open(kaggle_dir / 'kaggle.json', 'w') as f:
        json.dump(kaggle_json, f)
    
    # Set permissions
    os.chmod(kaggle_dir / 'kaggle.json', 0o600)
    
    print("Kaggle credentials set up successfully!")
    print("Please replace YOUR_KAGGLE_USERNAME and YOUR_KAGGLE_API_KEY")
    print("with your actual Kaggle username and API key.")

def main():
    setup_kaggle()

if __name__ == '__main__':
    main()
