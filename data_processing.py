import os
import pickle
import traceback
import cv2
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing
from functools import partial
from wavelet_utils import WaveMesh, process_image
from visualization import visualize_superpixels

from wavelet_utils import wavelet_superpixel, WaveMesh

# Preprocess image
def preprocess_image(image, sp_params):
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)  # Remove the channel dimension
    size = min(image.shape[:2])
    size = 2 ** int(np.log2(size))
    image = cv2.resize(image, (size, size))
    sp_dict = wavelet_superpixel(image, sp_params)
    return sp_dict

def create_data_object(graph):
    edges = list(graph.edges)
    edge_index = torch.tensor([item for edge in edges for item in edge], dtype=torch.long).view(2, -1).contiguous()
    edge_attr = torch.ones((edge_index.size(1), 1))
    x = torch.tensor([graph.nodes[n]['feature'] for n in graph.nodes], dtype=torch.float).view(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

""" def process_dataset_item(image_data, sp_params):
    try:
        image, label = image_data
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        #print(f"Squeezed image shape: {image.shape}")

        # Create WaveMesh instance
        wavemesh = WaveMesh(wname=sp_params.get('wname', 'db1'),
                            level=sp_params.get('level', None),
                            threshold_mult=sp_params.get('thresh_mult', 1.0))

        # Process the image using WaveMesh
        x, edge_index, edge_attr, superpixels, quadtree = process_image(image, wavemesh)
        
        #print(f"Superpixels shape: {superpixels.shape}")
        #print(f"x shape: {x.shape}")
        #print(f"edge_index shape: {edge_index.shape}")
        #print(f"edge_attr shape: {edge_attr.shape}")

        # Create the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long).view(1))

        return data
    except Exception as e:
        print(f"Error in process_dataset_item: {str(e)}")
        traceback.print_exc()
        return None """
def process_dataset_item(image_data, sp_params, idx=None):
    try:
        image, label = image_data
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        #print(f"Squeezed image shape: {image.shape}")

        wavemesh = WaveMesh(wname=sp_params.get('wname', 'db1'),
                            level=sp_params.get('level', None),
                            threshold_mult=sp_params.get('thresh_mult', 1.0))

        x, edge_index, edge_attr, superpixels, quadtree = process_image(image, wavemesh)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long).view(1),
                    superpixels=torch.from_numpy(superpixels))

        if idx is not None and idx < 5:
            os.makedirs('visualizations', exist_ok=True)
            visualize_superpixels(image, superpixels, f'visualizations/superpixels_{idx}.png')

        return data
    except Exception as e:
        print(f"Error in process_dataset_item: {str(e)}")
        traceback.print_exc()
        return None
        
""" def process_batch(batch, sp_params):
    return [process_dataset_item(image_data, sp_params) for image_data in batch] """
def process_batch(enumerated_batch, sp_params):
    batch_idx, batch = enumerated_batch
    return [process_dataset_item((image, label), sp_params, idx=i + batch_idx * len(batch)) 
            for i, (image, label) in enumerate(batch)]

""" def process_dataset_mp(dataset, sp_params, batch_size=64, num_workers=None, cache_dir='./cache'):
    if num_workers is None:
        num_workers = os.cpu_count()

    data_for_processing = [(image, label) for image, label in dataset]
    total_images = len(data_for_processing)
    
    batches = [data_for_processing[i:i + batch_size] for i in range(0, total_images, batch_size)]
    
    print(f"Processing {total_images} images in {len(batches)} batches using {num_workers} workers...")

    with multiprocessing.Pool(num_workers) as pool:
        process_func = partial(process_batch, sp_params=sp_params)
        results = list(tqdm(pool.imap(process_func, batches), total=len(batches), desc="Processing Batches", ncols=100))

    # Flatten the list of results
    graphs = [graph for batch_result in results for graph in batch_result if graph is not None]

    print(f"Finished processing. Total graphs created: {len(graphs)}")
    
    # Debug: Print information about the first few graphs
    for i, graph in enumerate(graphs[:5]):
        print(f"Graph {i}:")
        print(f"  x shape: {graph.x.shape}")
        print(f"  edge_index shape: {graph.edge_index.shape}")
        print(f"  edge_attr shape: {graph.edge_attr.shape if hasattr(graph, 'edge_attr') else 'No edge_attr'}")
        print(f"  y shape: {graph.y.shape}")

    return graphs """
def process_dataset_mp(dataset, sp_params, batch_size=64, num_workers=None, cache_dir='./cache'):
    if num_workers is None:
        num_workers = os.cpu_count()

    data_for_processing = [(image, label) for image, label in dataset]
    total_images = len(data_for_processing)
    
    batches = [data_for_processing[i:i + batch_size] for i in range(0, total_images, batch_size)]
    
    print(f"Processing {total_images} images in {len(batches)} batches using {num_workers} workers...")

    graphs = []
    with multiprocessing.Pool(num_workers) as pool:
        process_func = partial(process_batch, sp_params=sp_params)
        results = list(tqdm(pool.imap(process_func, enumerate(batches)), total=len(batches), desc="Processing Batches", ncols=100))

    # Flatten the list of results
    graphs = [graph for batch_result in results for graph in batch_result if graph is not None]

    print(f"Finished processing. Total graphs created: {len(graphs)}")
    
    return graphs

def load_processed_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def save_processed_graphs(graphs, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graphs, f)

def clear_cache(cache_dir='./cache'):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory {cache_dir} has been cleared and recreated.")

def calculate_avg_superpixels(dataset):
    total_superpixels = 0
    for data in dataset:
        total_superpixels += len(np.unique(data.superpixels)) - 1  # Subtract 1 to exclude background
    return total_superpixels / len(dataset)