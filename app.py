import torch
import os
import numpy as np
import cv2
from model import DnCNN 
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
import Class_predictor as run
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
import time
from torchvision import transforms
from PSNR_calc import calculate_psnr

timestamp = time.strftime(f"%Y.%m.%d-%H.%M.%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    lambda x: x.repeat(3, 1, 1) 
])
current_dir = os.getcwd()

def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ")
    return user_input.strip() or default_value

def normalize(img):
    return img / 255.0

def denormalize(img):
    img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def turn(input_image):
    input_image_denormalized = denormalize(input_image)  
    if len(input_image.shape) == 2: 
        input_image_uint8 = np.uint8(input_image_denormalized)
        return input_image_uint8
    elif len(input_image.shape) == 3: 
        input_image_uint8 = np.uint8(input_image_denormalized.transpose(1, 2, 0)) 
        return input_image_uint8
    else:
        raise ValueError("The output image has an unexpected shape.")

def save_as_png(output_dir, png_output_path, input_img, to_print=True):
    output_path = os.path.join(output_dir, png_output_path)
    cv2.imwrite(output_path, input_img)
    if(to_print):
        print(f"The image saved at ouput directory as: {png_output_path}")
    return output_path

def save_as_dicom(name, image_path, dicom_output_path, modality='OT'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    file_meta = Dataset()
    ds = FileDataset(dicom_output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = "Anonymous"
    ds.PatientID = "123456"
    ds.Modality = modality
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = ds.StudyDate
    ds.ContentDate = ds.StudyDate
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = ds.StudyTime
    ds.ContentTime = ds.StudyTime
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image.tobytes()

    ds.save_as(dicom_output_path)
    print(f"DICOM file in output directory as: {name}.dcm")

def delete_png(imag_path, to_print=True):
    if os.path.exists(imag_path):
        os.remove(imag_path)
        if (to_print):
            print(f"{imag_path} deleted successfully")
    else:
        print(f"{imag_path} not found")

def load_model(model_path, trainee__output, depth=17, device=device):
    model_path = os.path.join(model_path, 'model', 'model_checkpoint', 'model.pth')
    model = DnCNN(depth, 1).to(device)  
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"\nModel {trainee__output} loaded successfully!")
    return model

def model_eval(model, image_path, original_image_path, output_dir, level, device=device):
    input_image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError(f"The image at {image_path} could not be found.")
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    psnr_original_noisy = calculate_psnr(original_image_path, image_path)
    input_image_normalized = normalize(input_image_rgb)
    input_tensor = torch.from_numpy(input_image_normalized).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = output_tensor.squeeze().cpu().numpy()
    output_image_uint8 = turn(output_image)
    output_image_path=save_as_png(output_dir, f"DnCNN_denoised_output.png", output_image_uint8, to_print=False)
    psnr_denoised=calculate_psnr(original_image_path, output_image_path)
    os.remove(output_image_path)
    output_image_path=save_as_png(output_dir, f"DnCNN_denoised_output_Iteration_{level}_{psnr_denoised:.2f}.png", output_image_uint8)
    return psnr_denoised, output_image_path

def get_model(trainee__output, current_dir=current_dir):
    trainee_output=os.path.join(current_dir, trainee__output)
    if not os.path.exists(trainee_output):
        print("Path doesn't exist!!!\n")
        exit()
    return load_model(trainee_output, trainee__output)

output_folder = get_user_input(f"Enter the output directory name:", "output")
output_dir = os.path.join(current_dir, "Results", f"{output_folder}_{timestamp}")
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  
print("Output Directory: ", output_dir)
root = Tk()
root.withdraw() 
print(f'Select the image to be denoised...')
image_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

print(f'Select the original image...')
original_image_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)  
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

psnr_original_noisy = calculate_psnr(original_image_path, image_path)
noisy_image_path_2save=save_as_png(output_dir, f"Input_noisy_image_{psnr_original_noisy:.2f}.png", input_image)
original_image_path_2save=save_as_png(output_dir, f"Original_image.png", original_image)

time_start1=time.time()
model=get_model("train_data/e350-s55")
psnr_1, Iteration1_output= model_eval(model, image_path, original_image_path, output_dir, level=1)
time_end1=time.time()
Interative_1=cv2.imread(Iteration1_output, cv2.IMREAD_GRAYSCALE)

time_start2=time.time()
model=get_model("train_data/e300-s25_v1")
psnr_2, Iteration2_output= model_eval(model, Iteration1_output, original_image_path, output_dir, level=2)
time_end2=time.time()
Interative_2=cv2.imread(Iteration1_output, cv2.IMREAD_GRAYSCALE)
sel2='selected'
best_psnr=psnr_2
if(psnr_2<psnr_1):
    sel2='not_selected'
    Iteration2_output=Iteration1_output
    best_psnr=psnr_1
sel3='selected'

time_start3=time.time()
model=get_model("train_data/e100-s15")
psnr_3, Iteration3_output= model_eval(model, Iteration2_output, original_image_path, output_dir, level=3)
time_end3=time.time()
Interative_3=cv2.imread(Iteration1_output, cv2.IMREAD_GRAYSCALE)
print(f"SNR of the output (Iteration 1): {psnr_1:.2f} dB")
print(f"SNR of the output (Iteration 2): {psnr_2:.2f} dB")
print(f"SNR of the output (Iteration 3): {psnr_3:.2f} dB")
if(psnr_3<best_psnr):
    sel3='not_selected'
    Iteration3_output=Iteration2_output
DnCNN_Comparision = os.path.join(output_dir, "DnCNN_comparision.png")
plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.tight_layout()

plt.subplot(1, 5, 2)
plt.imshow(input_image, cmap='gray')
plt.title('Input Noisy Image')
plt.axis('off')
plt.tight_layout()

plt.subplot(1, 5, 3)
plt.imshow(Interative_1, cmap='gray')
plt.title(f'DnCNN Denoised Image \n(Iteration 1)\nSNR: {psnr_1:.2f}')
plt.axis('off')
plt.tight_layout()

plt.subplot(1, 5, 4)
plt.imshow(Interative_2, cmap='gray')
plt.title(f'DnCNN Denoised Image \n(Iteration 2 {sel2})\nSNR: {psnr_2:.2f}')
plt.axis('off')
plt.tight_layout()

plt.subplot(1, 5, 5)
plt.imshow(Interative_3, cmap='gray')
plt.title(f'DnCNN Denoised Image \n(Iteration 3 {sel3})\nSNR: {psnr_3:.2f}')
plt.axis('off')
plt.tight_layout()
plt.savefig(DnCNN_Comparision, dpi=300, bbox_inches='tight')
plt.show()

noisy_image_path =Iteration3_output
noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE) 

noisy_image_norm = noisy_image / 255.0
original_image_norm = original_image / 255.0

time_start =time.time()
prbf_denoised = cv2.bilateralFilter(noisy_image, d=6, sigmaColor=10, sigmaSpace=75) 
prbf_denoised = prbf_denoised / 255.0
prbf_denoised_resized = resize(prbf_denoised, original_image_norm.shape, anti_aliasing=True)
prbf_time=time.time() - time_start

time_start =time.time()
sigma_est = np.mean(estimate_sigma(noisy_image_norm))
hnipm_denoised = denoise_nl_means(noisy_image_norm, h=1.10 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11)
hnipm_denoised_resized = resize(hnipm_denoised, original_image_norm.shape, anti_aliasing=True)
hnipm_time=time.time()-time_start
noisy_image_resized = resize(noisy_image_norm, original_image_norm.shape, anti_aliasing=True)

output_image_path=save_as_png(output_dir, f"DnCNN_denoised_output.png", noisy_image, to_print=False)
output_image_uint8 = turn(hnipm_denoised_resized)
output_image_path_hnipm=save_as_png(output_dir, f'HNIPM_denoised_output.png', output_image_uint8, to_print=False)
output_image_uint8 = turn(prbf_denoised_resized)
output_image_path_prbf=save_as_png(output_dir, f'PFBF_denoised_output.png', output_image_uint8, to_print=False)

psnr_hnipm = calculate_psnr(original_image_path, output_image_path_hnipm)
psnr_prbf = calculate_psnr(original_image_path, output_image_path_prbf)
psnr_noisy = calculate_psnr(original_image_path, output_image_path)

# print("\n")
print(f"PSNR for DnCNN Periodic Denoising: {psnr_noisy:.2f} db")
print(f"PSNR for HNIPM denoising: {psnr_hnipm:.2f} dB")
print(f"PSNR for PRBF denoising: {psnr_prbf:.2f} dB")

# print("\n")
delete_png(output_image_path, to_print=False)
delete_png(output_image_path_hnipm, to_print=False)
delete_png(output_image_path_prbf, to_print=False)

print("\n")
output_image_path=save_as_png(output_dir, f"DnCNN_denoised_output_{psnr_noisy:.2f}.png", output_image_uint8)
output_image_uint8 = turn(hnipm_denoised_resized)
output_image_path_hnipm=save_as_png(output_dir, f'HNIPM_denoised_output_{psnr_hnipm:.2f}.png', output_image_uint8)
output_image_uint8 = turn(prbf_denoised_resized)
output_image_path_prbf=save_as_png(output_dir, f'PFBF_denoised_output_{psnr_prbf:.2f}.png', output_image_uint8)
Overall_comparision=os.path.join(output_dir, "Overall_compoarision.png")
fontsize = 9
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.title("(a) Original Image\n", fontdict={'fontsize': fontsize}, y=-0.2)

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')
plt.title(f"(b) Noisy Image\nPSNR: {psnr_noisy:.2f} dB", fontdict={'fontsize': fontsize}, y=-0.2)

plt.subplot(2, 2, 3)
plt.imshow(hnipm_denoised_resized, cmap='gray')
plt.axis('off')
plt.title(f"(c) Denoised using HNIPM Approximation \nPSNR: {psnr_hnipm:.2f} dB", fontdict={'fontsize': fontsize}, y=-0.2)

plt.subplot(2, 2, 4)
plt.imshow(prbf_denoised_resized, cmap='gray')
plt.axis('off')
plt.title(f"(d) Denoised using PRBF (Bilateral) Approximation \nPSNR: {psnr_prbf:.2f} dB", fontdict={'fontsize': fontsize}, y=-0.2)
plt.savefig(Overall_comparision)
plt.show()

print("\n")
save_as_dicom(f"DnCNN_denoised_output_{psnr_noisy:.2f}", output_image_path, os.path.join(output_dir, f"DnCNN_denoised_output_{psnr_noisy:.2f}.dcm"))
save_as_dicom(f"Input_noisy_image_{psnr_original_noisy:.2f}", noisy_image_path_2save, os.path.join(output_dir, f"Input_noisy_image_{psnr_original_noisy:.2f}.dcm"))
save_as_dicom(f"Original_image", original_image_path_2save, os.path.join(output_dir, f"Original_image.dcm"))
save_as_dicom(f"HNIPM_denoised_output_{psnr_hnipm:.2f}", output_image_path_hnipm,  os.path.join(output_dir, f'HNIPM_denoised_output_{psnr_hnipm:.2f}.dcm'))
save_as_dicom(f"PFBF_denoised_output_{psnr_prbf:.2f}", output_image_path_prbf, os.path.join(output_dir, f'PFBF_denoised_output_{psnr_prbf:.2f}.dcm'))

print("\nInference:")
print(f"Time take by the DnCNN iteration 1: {time_end1-time_start1:.4f}s | SNR value from the Iteration: {psnr_1:.2f} dB")
print(f"Time take by the DnCNN iteration 2: {time_end2-time_start2:.4f}s | SNR value from the Iteration: {psnr_2:.2f} dB")
print(f"Time take by the DnCNN iteration 3: {time_end3-time_start3:.4f}s | SNR value from the Iteration: {psnr_3:.2f} dB")
best_SNR1=psnr_1
if(best_SNR1<psnr_2):
    best_SNR=psnr_2
if(best_SNR1<psnr_3):
    best_SNR1=psnr_3
DnCNN_time=(time_end3-time_start1)
print(f"Total time taken by DnCNN: {DnCNN_time:.4f}s | Best SNR value from the Iterations: {best_SNR1:.2f} dB")
best_SNR2=psnr_hnipm
best_time=hnipm_time
if(psnr_hnipm<psnr_prbf):
    best_SNR2=psnr_prbf
    best_time=prbf_time
print(f"Time taken by HNIPM: {hnipm_time:.4f}s | SNR value from the HNIPM: {psnr_hnipm:.2f} dB")
print(f"Time taken by PRBF: {prbf_time:.4f}s  | SNR value from the HNIPM: {psnr_prbf:.2f} dB")
print(f"Time taken through the best computation: {DnCNN_time+best_time:.4f}s | Best SNR from computation: {best_SNR2:.2f} db")
print("\n")

categories1 = ['Iteration 1', 'Iteration 2', 'Iteration 3'] 
categories2 = ['DnCNN', 'HNIPM', 'PRBF']  
values1 = [psnr_1, psnr_2, psnr_3] 
values2 = [best_SNR1, psnr_hnipm, psnr_prbf]  
graph_path=os.path.join(output_dir, "SNR_graph.png")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(categories1, values1, marker='o', linestyle='-', color='b', label='Graph 1')
axs[0].set_xlabel("DnCNN iterations")
axs[0].set_ylabel("SNR in dB")
axs[0].set_title("Graph 1")
# axs[0].legend()
axs[0].grid()
for i, txt in enumerate(values1):
    axs[0].text(categories1[i], values1[i], f'{txt:.2f}', ha='right', va='bottom', fontsize=10, color='blue')

# Second graph
axs[1].plot(categories2, values2, marker='s', linestyle='--', color='r', label='Graph 2')
axs[1].set_xlabel("Overall computing")
axs[1].set_ylabel("SNR in dB")
axs[1].set_title("Graph 2")
# axs[1].legend()
axs[1].grid()
for i, txt in enumerate(values2):
    axs[1].text(categories2[i], values2[i], f'{txt:.2f}', ha='left', va='top', fontsize=10, color='red')

plt.tight_layout()
plt.savefig(graph_path)
plt.show()

AI_send=output_image_path_hnipm
if psnr_hnipm<psnr_prbf:
    AI_send=output_image_path_prbf
run.upload_image(output_dir, AI_send)

print(f"Interpretation written to {output_dir}\log.txt")
print("Thank you!!!\n")
exit()