import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
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

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = PascalVOCDataset("./train", "./train", transform)
test_dataset = PascalVOCDataset("./test", "./test", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load ResNet-50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save model if loss improves
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "resnet50_best_model.pth")
        print("Model saved with improved loss.")

print("ResNet-50 Training Completed")

