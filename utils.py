import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset

def visualize_superpixels(image, superpixels, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(superpixels, cmap='nipy_spectral')
    plt.title('Superpixels')
    plt.savefig(save_path)
    plt.close()

def process_subset(dataset, wavemesh, num_samples=500):
    processed_data = []
    superpixel_counts = []
    
    subset_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, subset_indices)
    
    for i, (image, label) in enumerate(tqdm(subset, desc="Processing subset")):
        superpixels, quadtree = wavemesh(image.squeeze().numpy())
        processed_data.append((superpixels, quadtree, label))
        superpixel_counts.append(len(np.unique(superpixels)))
        
        if i < 5:  # Visualize first 5 images
            visualize_superpixels(image, superpixels, f'superpixels_{i}.png')
        
        if i % 100 == 0 and i > 0:
            print(f"Processed {i} images. Avg superpixels: {np.mean(superpixel_counts):.2f}")
    
    return processed_data, superpixel_counts