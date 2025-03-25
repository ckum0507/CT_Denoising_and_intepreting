import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define dataset class to parse Pascal VOC XML annotations
class PascalVOCDataset(Dataset):
    def __init__(self, img_dir, xml_dir, transform=None):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.classes = ["Aneurysm", "Cancer", "Normal_Brain", "Tumour-Glioma", "Tumour-Meningioma", "Tumour-Pituitary"]
    
    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        labels = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name in self.classes:
                labels.append(self.classes.index(class_name))
        return labels[0] if labels else 0  # Default to Normal_Brain if empty
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace('.jpg', '.xml'))
        img = Image.open(img_path).convert("RGB")
        label = self.parse_xml(xml_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define data transformations - using smaller image size
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Smaller input size to reduce memory usage
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = PascalVOCDataset("./train", "./train", transform)
val_dataset = PascalVOCDataset("./test", "./test", transform)
test_dataset = PascalVOCDataset("./test", "./test", transform)

# Use smaller batch size
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a smaller model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(train_dataset.classes))
model = model.to(device)

# Enable mixed precision training if on GPU
use_amp = device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 40
best_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Use mixed precision training if on GPU
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scale gradients and optimize
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # Save model if validation loss improves
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, "efficientnet_best_model.pth")
        print("Model saved with improved validation loss.")

print("EfficientNet Training Completed")

# Clear GPU memory
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Test the model
print("Testing the model...")
model.eval()
test_correct = 0
test_total = 0
class_correct = [0] * len(train_dataset.classes)
class_total = [0] * len(train_dataset.classes)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            class_correct[label] += (pred == label).item()
            class_total[label] += 1

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Print per-class accuracy
for i in range(len(train_dataset.classes)):
    class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"Accuracy of {train_dataset.classes[i]}: {class_acc:.2f}%")