import torch
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from PIL import Image
import os
import torch.nn.functional as F  
import time

CLASS_NAMES = ["Aneurysm", "Cancer", "Normal Brain", "Tumor-Glioma", "Tumor-Meningioma", "Tumor-Pituitary"]

def load_model(model_name, model_path, num_classes=6):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b7":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes) 

    else:
        raise ValueError("Invalid model name")

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()  
        return model
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None  

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file not found at {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")  
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    if model is None:
        return "Model not loaded", 0.0, 0.0  

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)  
        confidence, predicted_idx = torch.max(probabilities, 1)  

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    predicted_confidence = confidence.item() * 100  
    return predicted_class, predicted_confidence

def upload_image(output_dir, image_path):

    resnet50_path = "./interpret/resnet50.pth" 
    efficientnet_b7_path = "./interpret/efficientnet-b7.pth" 

    resnet50_model = load_model("resnet50", resnet50_path)
    efficientnet_model = load_model("efficientnet_b7", efficientnet_b7_path)

    if resnet50_model is not None:
        print("ResNet-50 Model Loaded Successfully.")
    else:
        print("Failed to load ResNet-50.")
        exit()
    if efficientnet_model is not None:
        print("EfficientNet-B7 Model Loaded Successfully.")
    else:
        print("Failed to load EfficientNet-B7.")
        exit()

    log_file_path = os.path.join(output_dir, "log.txt")
    with open(log_file_path, "w") as file:
        file.write("Inference log:\n")
        file.close()
    image_tensor = preprocess_image(image_path)
    time_start1 = time.time()
    resnet_prediction, resnet_confidence = predict(resnet50_model, image_tensor)
    time_end1 = time.time()
    end1 = time_end1 - time_start1
    time_start2 = time.time()
    efficientnet_prediction, efficientnet_confidence = predict(efficientnet_model, image_tensor)
    time_end2 = time.time()
    end2 = time_end2 - time_start2
    with open(log_file_path, "a") as file:
        output=f"ResNet-50 Prediction: {resnet_prediction} | Confidence: {resnet_confidence:.2f}% | Time taken: {end1:.2f} sec\n"
        print(output.strip()) 
        file.write(output) 
        output=f"EfficientNet-B7 Prediction: {efficientnet_prediction} | Confidence: {efficientnet_confidence:.2f}% | Time Taken: {end2:.2f} sec"
        print(output.strip()) 
        file.write(output) 
