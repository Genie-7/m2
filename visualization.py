# visualization.py
import matplotlib.pyplot as plt
from scipy.stats import describe
import numpy as np

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

def analyze_graphs(graphs):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.num_edges // 2 for g in graphs]  # Divide by 2 because edges are bidirectional
    
    # Filter out graphs with zero nodes
    valid_graphs = [(n, e) for n, e in zip(num_nodes, num_edges) if n > 0]
    if not valid_graphs:
        print("Error: No valid graphs found (all graphs have zero nodes)")
        return
    
    valid_num_nodes, valid_num_edges = zip(*valid_graphs)
    avg_degree = [e / n for n, e in valid_graphs]
    
    print("Graph Statistics:")
    print(f"Total graphs: {len(graphs)}")
    print(f"Valid graphs: {len(valid_graphs)}")
    print(f"Graphs with zero nodes: {len(graphs) - len(valid_graphs)}")
    print(f"Number of nodes: {describe(valid_num_nodes)}")
    print(f"Number of edges: {describe(valid_num_edges)}")
    print(f"Average degree: {describe(avg_degree)}")

    # Additional analysis
    zero_edge_graphs = sum(1 for e in num_edges if e == 0)
    print(f"Graphs with zero edges: {zero_edge_graphs}")

    # Histogram of number of nodes
    hist, bin_edges = np.histogram(valid_num_nodes, bins=10)
    print("\nHistogram of number of nodes:")
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {hist[i]}")