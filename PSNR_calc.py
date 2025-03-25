import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def calculate_psnr(original, variant):
    original_img = cv2.imread(original)
    variant_img = cv2.imread(variant)
    
    if original_img is None or variant_img is None:
        return None
    
    variant_img = cv2.resize(variant_img, (original_img.shape[1], original_img.shape[0]))

    mse = np.mean((original_img - variant_img) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def open_file(index, file_paths, labels):
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if filepath:
        file_paths[index] = filepath
        img = Image.open(filepath)
        img.thumbnail((256, 256))
        img = ImageTk.PhotoImage(img)
        labels[index].config(image=img)
        labels[index].image = img

def compute_psnr(file_paths, number, result_label):
    if not file_paths[0]:
        messagebox.showerror("Error", "Please select the original image.")
        return
    
    results = []
    for i in range(1, number):
        if file_paths[i]:
            psnr_value = calculate_psnr(file_paths[0], file_paths[i])
            results.append(f"Variant {i}: {psnr_value:.2f} dB" if psnr_value else f"Variant {i}: Error")
        else:
            results.append(f"Variant {i}: Not selected")
    
    result_label.config(text="\n".join(results))

def computer():
    number = int(input("Enter the number of variants for the original image: ")) + 1
    
    root = tk.Tk()
    root.title("PSNR Calculator")
    
    file_paths = [None] * number
    labels = []
    
    for i in range(number):
        btn_text = "Select Original Image" if i == 0 else f"Select Variant {i}"
        btn = tk.Button(root, text=btn_text, command=lambda idx=i: open_file(idx, file_paths, labels))
        btn.grid(row=i, column=0)
        
        label = tk.Label(root, text="No Image", width=12, height=6, relief=tk.SUNKEN)
        label.grid(row=i, column=1)
        labels.append(label)
    
    result_label = tk.Label(root, text="", justify=tk.LEFT)
    result_label.grid(row=number + 1, column=0, columnspan=2)
    
    compute_btn = tk.Button(root, text="Compute PSNR", command=lambda: compute_psnr(file_paths, number, result_label))
    compute_btn.grid(row=number, column=0, columnspan=2)
    
    root.mainloop()

