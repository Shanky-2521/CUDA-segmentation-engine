import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path

# Import config and model
from src.config import Config
from src.model.deeplabv3 import DeepLabV3

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Get image and mask paths
        self.image_paths = sorted(list((self.data_dir / split / 'img').glob('*.png')))
        self.mask_paths = sorted(list((self.data_dir / split / 'label').glob('*.png')))
        
        # Verify pairs
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item at index"""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask

class MaskToLong(A.DualTransform):
    """Convert mask to torch.long type"""
    
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img, **params):
        return img
    
    def apply_to_mask(self, mask, **params):
        return mask.to(torch.long)
    
    def get_transform_init_args_names(self):
        return ()


def get_transforms(split):
    """Get data transforms for training and validation"""
    if split == 'train':
        return A.Compose([
            A.Resize(height=512, width=1024, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
            MaskToLong()
        ])
    else:
        return A.Compose([
            A.Resize(height=512, width=1024, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
            MaskToLong()
        ])



def collate_fn(batch):
    """Custom collate function to handle masks"""
    # Filter out None values from batch
    batch = [(img, mask) for img, mask in batch if img is not None and mask is not None]
    
    if not batch:  # If empty after filtering
        return None, None
    
    images, masks = zip(*batch)
    
    # Convert to tensors if not already
    images = [torch.tensor(img) if not torch.is_tensor(img) else img for img in images]
    masks = [torch.tensor(msk) if not torch.is_tensor(msk) else msk for msk in masks]
    
    # Stack images
    images = torch.stack(images)
    
    # Stack masks and ensure they are in the correct format
    masks = torch.stack(masks)
    
    # Ensure masks have the correct shape for CrossEntropyLoss (N, H, W)
    if masks.dim() == 4:  # If masks have an extra channel dimension
        masks = masks.squeeze(1)
    
    # Print shapes for debugging
    print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
    
    return images, masks

class Trainer:
    def __init__(self, config):
        """Initialize trainer with configuration"""
        self.config = config
        
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError("CUDA is not available on this system. Please ensure CUDA is properly installed.")
        
        # Initialize model
        self.model = DeepLabV3(num_classes=self.config.NUM_CLASSES)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Initialize data loaders
        self.create_data_loaders()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        train_dataset = SegmentationDataset(
            self.config.DATA_DIR,
            split='train',
            transform=get_transforms('train')
        )
        val_dataset = SegmentationDataset(
            self.config.DATA_DIR,
            split='val',
            transform=get_transforms('val')
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
    def train_one_epoch(self, epoch):
        """Train model for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            try:
                # Skip if batch is None
                if images is None or masks is None:
                    print(f"Skipping batch {batch_idx} due to None values")
                    continue
                    
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
                # Forward pass with automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Detach tensors to prevent memory leaks
                images = images.detach()
                masks = masks.detach()
                outputs = outputs.detach()
                
                # Print statistics
                running_loss += loss.item()
                if batch_idx % self.print_freq == self.print_freq - 1:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                          f'Step [{batch_idx + 1}/{len(self.train_loader)}], '
                          f'Loss: {running_loss / self.print_freq:.4f}')
                    running_loss = 0.0
                    
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"CUDA out of memory in batch {batch_idx}, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                raise
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return running_loss / len(self.train_loader)

    def validate(self):
        """Validate model on validation set"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                try:
                    if images is None or masks is None:
                        print("Skipping validation batch due to None values")
                        continue
                        
                    # Move data to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        
                    running_loss += loss.item()
                    
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Detach tensors
                    images = images.detach()
                    masks = masks.detach()
                    outputs = outputs.detach()
                    
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("CUDA out of memory in validation, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    raise
                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        
        return running_loss / len(self.val_loader)
        
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, f'models/checkpoint_epoch_{epoch}.pth')
        
    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print('Best model saved!')
            
            print('-' * 50)

def main():
    # Create config instance
    config = Config()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Verify dataset structure
    print("\nVerifying dataset structure...")
    train_dir = Path(config.DATA_DIR) / 'train' / 'img'
    val_dir = Path(config.DATA_DIR) / 'val' / 'img'
    
    # Check training data
    print("\nTraining data:")
    print(f"Number of training images: {len(list(train_dir.glob('*.png')))}")
    
    # Check validation data
    print("\nValidation data:")
    print(f"Number of validation images: {len(list(val_dir.glob('*.png')))}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train(num_epochs=100)

if __name__ == '__main__':
    main()
