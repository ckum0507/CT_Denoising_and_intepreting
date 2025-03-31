import os
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.getcwd()
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_dir = os.path.join(current_dir, "Brain_data_set")
dataset = datasets.ImageFolder(data_dir, transform=transformer)

num_classes = len(dataset.classes) 
resnet_model = models.resnet50(weights="IMAGENET1K_V1")  
in_features_resnet = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(in_features_resnet, num_classes)
resnet_model=resnet_model.to(device)

efficientnet_model = models.efficientnet_b7(weights="IMAGENET1K_V1") 
in_features_efficientnet = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(in_features_efficientnet, num_classes)
efficientnet_model=efficientnet_model.to(device)
# def modify_checkpoint(path, model):
#     checkpoint = torch.load(path, weights_only=True)
#     if "classifier.1.weight" in checkpoint:
#         del checkpoint["classifier.1.weight"]
#     if "classifier.1.bias" in checkpoint:
#         del checkpoint["classifier.1.bias"]
#     torch.save(checkpoint, path.replace(".pth", "_modified.pth"))
#     model.load_state_dict(torch.load(path.replace(".pth", "_modified.pth"), weights_only=True), strict=False)

def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")  
    img = transformer(img).unsqueeze(0).to(device)  

    with torch.no_grad():  
        start_time = time.time() 
        output = model(img)
        _, predicted = torch.max(output, 1)
        prediction_time = time.time() - start_time 
        predicted_class = dataset.classes[predicted.item()]  

    return predicted_class, prediction_time

def evaluate_model(model, model_name, image_path, log_file):
    prediction, prediction_time = predict_image(model, image_path)
    output = f"Model: {model_name}, Predicted Class: {prediction}, Inference Time: {prediction_time:.4f}s\n"
    print(output.strip()) 
    log_file.write(output) 

def upload_image(output_dir, file_path):
    if not output_dir:
        os.makedirs(output_dir)
    log_file_path = os.path.join(output_dir, "log.txt")

    with open(log_file_path, "w") as file:
        file.write("Inference log:\n")
        file.close()
    with open(log_file_path, "a") as file:
        evaluate_model(resnet_model, "ResNet-50", file_path, file)
        evaluate_model(efficientnet_model, "EfficientNet-B7", file_path, file)
        file.close()