import os
import torch
import matplotlib.pyplot as plt
import tarfile
from PIL import Image
from torchvision import transforms
from model import DnCNN
import nntools as nt
from utils import DenoisingStatsManager, plot
from torch.utils.data import Dataset
from torchvision import transforms
import shutil
import cv2
import numpy as np


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    lambda x: x.repeat(3, 1, 1) 
])
class NoisyDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, image_size=None):

        self.noisy_dir = os.path.join(root_dir, mode, 'noisy')
        self.clear_dir = os.path.join(root_dir, mode, 'clear')
        self.noisy_files = sorted(os.listdir(self.noisy_dir))
        self.clear_files = sorted(os.listdir(self.clear_dir))
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clear_path = os.path.join(self.clear_dir, self.clear_files[idx])

        noisy_image = Image.open(noisy_path).convert('L')
        clear_image = Image.open(clear_path).convert('L')

        if self.image_size:
            noisy_image = noisy_image.resize(self.image_size)
            clear_image = clear_image.resize(self.image_size)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clear_image = self.transform(clear_image)

        return noisy_image, clear_image

def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ")
    return user_input.strip() or default_value

def save_as_png(output_dir, png_output_path, input_img):
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.cpu().detach().numpy()
    input_img = np.uint8(input_img * 255)
    if input_img.ndim == 2:
        output_path = os.path.join(output_dir, png_output_path)
        cv2.imwrite(output_path, input_img)
    elif input_img.ndim == 3 and input_img.shape[0] == 3:
        input_img = input_img.transpose(1, 2, 0)
        output_path = os.path.join(output_dir, png_output_path)
        cv2.imwrite(output_path, input_img)
    else:
        raise ValueError(f"Invalid image shape: {input_img.shape}. Image must be either 2D (grayscale) or 3D (RGB).")

    print(f"The image saved at output directory as: {png_output_path}")

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.getcwd()

root_dir = get_user_input("Enter the root directory for the dataset", rf"{current_dir}\Denoising_DnCNN_dataset")
image_name = get_user_input("Enter the image name", "cd.jpg")

image_size = (512, 512)
test_image_size = (512, 512)
sigma = float(get_user_input("Enter sigma level", 25))
model_type = "dncnn"
depth = 50
channels = 1 
learning_rate = float(get_user_input("Enter Learning rate (<0.01)", 0.001))
batch_size = int(get_user_input("Enter Batch size", 64))
num_epochs = int(get_user_input("Enter number of epochs directory", 300))
output__dir = get_user_input("Enter output directory", rf"{current_dir}\trainee_output\trainee_output_{sigma}_{num_epochs}")
output_dir= os.path.join(output__dir, "model")
os.makedirs(output_dir, exist_ok=True)
plot_results = True

train_set = NoisyDataset(root_dir, mode="train", transform=transform, image_size=image_size)
test_set = NoisyDataset(root_dir, mode="test", transform=transform, image_size=test_image_size)

net = DnCNN(depth, C=channels).to(device)

adam = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)

stats_manager = DenoisingStatsManager()

checkpoint_path = os.path.join(output_dir, "model_checkpoint")
if os.path.exists(checkpoint_path):
    print("Checkpoint detected.")
    user_choice = get_user_input("A conflicting checkpoint was found. Do you want to continue? (yes/no)", "yes").strip().lower()
    if user_choice == 'yes':
        print("Resuming training with the current checkpoint.")
        resume_training = True
    else:
        shutil.rmtree(output_dir) 
        print("Checkpoint cleared. Starting fresh.")
        resume_training = False        
else:
    resume_training = False

exp = nt.Experiment(
    net, train_set, test_set, adam, stats_manager,
    batch_size=batch_size, output_dir=output_dir,
    perform_validation_during_training=True,
    force_new_experiment=False
)

if resume_training:
    try:
        exp.load() 
        print(f"Resuming checkpoint...")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1) 

try:
    if plot_results:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(num_epochs=num_epochs, plot=lambda exp: plot(exp, fig=fig, axes=axes, noisy=test_set[73][0]))
    else:
        exp.run(num_epochs=num_epochs)
except KeyboardInterrupt as e:
    print(f"Training halted due to KeyboardInterrupt")

model_folder = os.path.join(output_dir, "model_checkpoint")
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "model.pth")
torch.save(net.state_dict(), model_path)

metadata = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "sigma": sigma,
    "model_type": model_type,
}
metadata_path = os.path.join(model_folder, "metadata.pth")
torch.save(metadata, metadata_path)

tar_path = os.path.join(output_dir, "model_checkpoint.tar")
with tarfile.open(tar_path, "w") as tar:
    tar.add(model_folder, arcname=os.path.basename(model_folder))
print(f"Model and metadata saved loaction: {output_dir}")
print(f"Model and metadata saved as model_checkpoint.tar")

image_path = os.path.join(root_dir, image_name)
image = Image.open(image_path).convert('L') 

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to(device) 

image_tensor = image_tensor.repeat(1, 3, 1, 1) 

with torch.no_grad():
    denoised_image = net(image_tensor).cpu().squeeze(0) 

denoised_image = torch.clamp(denoised_image, 0, 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0), cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0), cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0), cmap='gray') 
plt.title('Ground Truth')
plt.axis('off')

img_output_dir=os.path.join(output__dir, "output_images")
os.makedirs(img_output_dir, exist_ok=True)
print(f"Image outputs saved at: {img_output_dir}")
save_as_png(img_output_dir, "Noisy Image.png", image_tensor.squeeze(0).cpu().numpy())
save_as_png(img_output_dir, f"Denoised Image.png", denoised_image.squeeze(0).cpu().numpy())
plot_path = os.path.join(img_output_dir, "training_plot.png")
fig.savefig(plot_path, format='png')
print(f"Plot saved as: training_plot.png")

plt.tight_layout()
plt.show(block=False)
plt.pause(60)
plt.close()