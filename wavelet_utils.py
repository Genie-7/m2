import numpy as np
from skimage.segmentation import relabel_sequential
import pywt
import networkx as nx
from skimage.measure import regionprops
from scipy import ndimage
import torch

# WaveMesh implementation
class WaveMesh:
    def __init__(self, wname='db1', level=None, threshold_mult=1.0):
        self.wname = wname
        self.level = level
        self.threshold_mult = threshold_mult
        self.original_shape = None

    def decompose(self, image):
        # Ensure image is 2D and numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        # Determine the maximum decomposition level
        if self.level is None:
            self.level = min(pywt.dwt_max_level(min(image.shape), self.wname), 5)  # Increase max level
        else:
            self.level = min(self.level, pywt.dwt_max_level(min(image.shape), self.wname))

        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wname, level=self.level)
        return coeffs

    def threshold_coeffs(self, coeffs):
        # Calculate the threshold
        threshold = self.threshold_mult * np.sqrt(2 * np.log(np.prod(coeffs[0].shape)))

        # Reduce the threshold to produce more superpixels
        threshold *= 0.1  # Adjust this factor as needed

        # Apply thresholding
        thresholded = [coeffs[0]]  # Keep the approximation coefficients
        for detail_coeffs in coeffs[1:]:
            thresholded.append(tuple(pywt.threshold(d, threshold, mode='soft') for d in detail_coeffs))
        return thresholded

    def create_quadtree(self, coeffs):
        quadtree = np.zeros(coeffs[0].shape, dtype=int)
        for level in range(1, len(coeffs)):
            level_size = coeffs[0].shape[0] // (2**(level-1))
            for i in range(3):
                detail = np.abs(coeffs[level][i]) > 0
                if detail.shape != (level_size, level_size):
                    # Resize detail to match level_size
                    detail = np.repeat(np.repeat(detail, 2**(level-1), axis=0), 2**(level-1), axis=1)
                    detail = detail[:level_size, :level_size]
                quadtree[:level_size, :level_size] += detail * (2**i)
        
        # Ensure quadtree is at least 2x2
        if quadtree.shape[0] < 2 or quadtree.shape[1] < 2:
            quadtree = np.zeros((2, 2), dtype=int)
        
        return quadtree

    def generate_superpixels(self, quadtree):
        if quadtree.ndim == 0:  # If quadtree is a single value
            return np.zeros(self.original_shape, dtype=int)
        
        superpixels = np.zeros(self.original_shape, dtype=int)
        label = 1
        min_size = 2  # Reduce this to allow for smaller superpixels
        for level in range(self.level, 0, -1):
            size = max(2**level, min_size)
            for i in range(0, self.original_shape[0], size):
                for j in range(0, self.original_shape[1], size):
                    qi = min(i // (self.original_shape[0] // quadtree.shape[0]), quadtree.shape[0] - 1)
                    qj = min(j // (self.original_shape[1] // quadtree.shape[1]), quadtree.shape[1] - 1)
                    if quadtree[qi, qj] > 0:
                        superpixels[i:min(i+size, self.original_shape[0]), j:min(j+size, self.original_shape[1])] = label
                        label += 1
        return superpixels

    def __call__(self, image):
        #print(f"Input image shape to WaveMesh: {image.shape}")
        self.original_shape = image.shape
        coeffs = self.decompose(image)
        #print(f"Number of coefficient levels: {len(coeffs)}")
        #for i, c in enumerate(coeffs):
            #print(f"Level {i} shape: {c[0].shape if i > 0 else c.shape}")
        thresholded_coeffs = self.threshold_coeffs(coeffs)
        quadtree = self.create_quadtree(thresholded_coeffs)
        #print(f"Quadtree shape: {quadtree.shape}")
        
        if quadtree.size == 1:
            #print("Warning: Quadtree has been reduced to a single value. Adjusting...")
            quadtree = np.zeros((2, 2), dtype=int)
        
        superpixels = self.generate_superpixels(quadtree)
        #print(f"Superpixels shape: {superpixels.shape}")
        return superpixels, quadtree
    
def process_image(image, wavemesh):
    superpixels, quadtree = wavemesh(image)
    #print(f"Superpixels unique values: {np.unique(superpixels)}")
    graph = construct_graph(superpixels, image)
    x, edge_index, edge_attr = graph_to_sparse(graph)
    return x, edge_index, edge_attr, superpixels, quadtree

def construct_graph(superpixels, image):
    G = nx.Graph()
    regions = regionprops(superpixels)

    # Convert image to numpy if it's a PyTorch tensor
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # Add nodes
    for region in regions:
        G.add_node(region.label, centroid=region.centroid,
                   mean_intensity=np.mean(image[superpixels == region.label]))

    # Add edges
    for region in regions:
        for neighbor in regions:
            if region.label != neighbor.label:
                if np.any(np.abs(np.array(region.coords)[:, None, :] - neighbor.coords) <= 1):
                    centroid_diff = np.array(region.centroid) - np.array(neighbor.centroid)
                    G.add_edge(region.label, neighbor.label, weight=np.linalg.norm(centroid_diff))

    return G

def graph_to_sparse(G):
    num_nodes = len(G.nodes)
    edge_index = []
    edge_attr = []
    x = []

    for node in G.nodes(data=True):
        x.append([node[1]['mean_intensity']])

    for edge in G.edges(data=True):
        edge_index.append([edge[0]-1, edge[1]-1])
        edge_index.append([edge[1]-1, edge[0]-1])  # Add reverse edge for undirected graph
        # Ensure edge_attr is 2-dimensional
        attr = [edge[2]['weight'], 0.0]  # Add a second dimension with a default value
        edge_attr.append(attr)
        edge_attr.append(attr)  # Repeat for reverse edge

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr


def cantor_pairing(a, b):
    return (a + b) * (a + b + 1) // 2 + b

def one_time_filter(img_var, coeffs, thresh_val, verbose=False):
    coeffs_thresh = pywt.threshold(coeffs, thresh_val, mode='hard', substitute=0)
    fimg_nnz = np.count_nonzero(coeffs_thresh)
    fimg_var = (coeffs_thresh**2).mean()
    return coeffs_thresh, fimg_nnz, fimg_var

def recursive_filter(img_var, coeffs, verbose=False):
    noise_var = img_var
    N = coeffs.shape[0] * coeffs.shape[1]
    rel_tol = 0.1e-2
    thresh_old = 0.0
    thresh_new = np.sqrt(2.0 * np.log(N) * noise_var)

    while np.abs(thresh_new - thresh_old) > rel_tol * thresh_old:
        thresh_old = thresh_new
        coeffs_thresh = pywt.threshold(coeffs, thresh_old, mode='hard', substitute=0)
        noise = coeffs - coeffs_thresh
        noise_var = (noise**2).mean()
        thresh_new = np.sqrt(2.0 * np.log(N) * noise_var)

    fimg_nnz = np.count_nonzero(coeffs_thresh)
    fimg_var = (coeffs_thresh**2).mean()

    return coeffs_thresh, thresh_old, fimg_nnz, fimg_var

def create_mesh(Nx, rcoeffs, thresh_val, max_scale, verbose=False):
    mask = [np.empty((1))]
    for s in range(1, max_scale + 1):
        mask.append(np.logical_or(np.logical_or(abs(rcoeffs[s][0]) > thresh_val, 
                                                abs(rcoeffs[s][1]) > thresh_val),
                                  abs(rcoeffs[s][2]) > thresh_val))
    for s in range(max_scale, 1, -1):
        for i in range(0, mask[s].shape[0]):
            for j in range(0, mask[s].shape[1]):
                if mask[s][i, j]:
                    mask[s - 1][i // 2, j // 2] = True
    
    sp = np.zeros((Nx, Nx), dtype=np.int64)
    tmp = 0
    for s in range(1, max_scale + 1):
        delta = int(Nx / 2**(s - 1))
        for i in range(0, mask[s].shape[0]):
            for j in range(0, mask[s].shape[1]):
                if mask[s][i, j]:
                    lft = np.array([delta * i, delta * j])
                    ctr_lft = lft + int(delta / 2) - 1
                    ctr_rgt = ctr_lft + 1
                    rgt_exc = lft + delta
                    sp[lft[0]:1 + ctr_lft[0], lft[1]:1 + ctr_lft[1]] = tmp
                    sp[lft[0]:1 + ctr_lft[0], ctr_rgt[1]:rgt_exc[1]] = tmp + 1
                    sp[ctr_rgt[0]:rgt_exc[0], lft[1]:1 + ctr_lft[1]] = tmp + 2
                    sp[ctr_rgt[0]:rgt_exc[0], ctr_rgt[1]:rgt_exc[1]] = tmp + 3
                    tmp += 4

    sp, *__ = relabel_sequential(sp + 1)
    return sp

def is_power_of_two(n):
    return n & (n - 1) == 0

def check_img(img):
    if len(img.shape) == 3 and img.shape[0] == 1:  # Single-channel 3D image
        img = img.squeeze(0)  # Remove the channel dimension
    assert(len(img.shape) == 2), f"Expected 2D image, got shape {img.shape}"
    assert(img.shape[0] == img.shape[1]), f"Image must be square, got shape {img.shape}"
    assert(is_power_of_two(img.shape[0])), f"Image size must be a power of 2, got {img.shape[0]}"

def wavelet_superpixel_singlechannel(image, params):
    wname = params['wname']
    num_sp = params['number']
    thresh_mult = params['thresh_mult']
    verbose = params['verbose']

    img = image
    Nx, Ny = img.shape

    img_mean = np.mean(img)
    img -= img_mean
    coeffs = pywt.wavedec2(img, wname)
    coeffs_arr_2d, coeff_slices = pywt.coeffs_to_array(coeffs)

    img_var = ((img-np.mean(img))**2).mean()

    if num_sp == 0:  # use automatic algorithm
        coeffs_arr_2d_thresh, thresh_val, fimg_nnz, fimg_var = recursive_filter(
            img_var, coeffs_arr_2d, verbose=verbose)

        if thresh_mult != 1:
            if not thresh_val > 0:
                print('*** wavelet threshold is 0! ***')
            thresh_val *= thresh_mult
            coeffs_arr_2d_thresh, fimg_nnz, fimg_var = one_time_filter(
                img_var, coeffs_arr_2d, thresh_val, verbose=verbose)
    else:
        coeffs_arr_2d_raveled = coeffs_arr_2d.ravel()
        coeffs_arr_2d_raveled_sorted = np.sort(coeffs_arr_2d_raveled)
        thresh_val = coeffs_arr_2d_raveled_sorted[-num_sp//4-1]
        coeffs_arr_2d_thresh, fimg_nnz, fimg_var = one_time_filter(
            img_var, coeffs_arr_2d, thresh_val, verbose=verbose)

    rcoeffs = pywt.array_to_coeffs(coeffs_arr_2d_thresh, coeff_slices, 'wavedec2')

    sp_dict = {}
    for scale in range(1, len(rcoeffs)):  # 1 is coarsest
        sp_dict[scale] = create_mesh(Nx, rcoeffs, 0., scale, verbose=verbose)

    return sp_dict

def wavelet_superpixel(image, sp_params):
    # Use a simpler wavelet transform
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    
    # Threshold the coefficients
    thresh = np.mean(np.abs(coeffs[-1][0])) * sp_params['thresh_mult']
    coeffs_thresholded = [coeffs[0]] + [
        (pywt.threshold(d[0], thresh, mode='hard'),
         pywt.threshold(d[1], thresh, mode='hard'),
         pywt.threshold(d[2], thresh, mode='hard'))
        for d in coeffs[1:]
    ]
    
    # Reconstruct the image
    reconstructed = pywt.waverec2(coeffs_thresholded, 'haar')
    
    # Create superpixels based on the reconstructed image
    sp = np.zeros_like(image, dtype=int)
    sp[reconstructed > thresh] = 1
    
    return {1: sp}, image.shape

