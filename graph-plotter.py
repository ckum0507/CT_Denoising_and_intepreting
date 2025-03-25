import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def plot_accuracy_from_file():
    root = tk.Tk()
    root.withdraw() 
    filename = filedialog.askopenfilename(title="Select a text file", filetypes=[("Text Files", "*.txt")])
    if not filename:
        print("No file selected")
        return
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    base_title = os.path.basename(filename).split('.')[0]  
    title = base_title + " - Training Accuracy vs Validation Accuracy"
    output_file = title + ".png" 
    
    epochs = []
    train_acc = []
    val_acc = []
    
    for line in lines[1:]:
        parts = line.split(',')
        epoch = int(parts[0].split('[')[1].split('/')[0])
        train_accuracy = float(parts[2].split(':')[1].strip())
        val_accuracy = float(parts[3].split(':')[1].strip())
        
        epochs.append(epoch)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker='o', linestyle='-', color='blue', label='Train Acc')
    plt.plot(epochs, val_acc, marker='s', linestyle='-', color='orange', label='Val Acc')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.xticks(range(0, 13, 1)) 
    plt.yticks([i / 10 for i in range(0, 11)])  
    plt.xlim(0, 12)
    plt.ylim(0, 1)
    
    plt.savefig(output_file) 
    plt.show()


plot_accuracy_from_file()
plot_accuracy_from_file()
plot_accuracy_from_file()