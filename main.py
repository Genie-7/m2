import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
from torchvision.datasets import MNIST
import traceback
import pickle
import os
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from visualization import analyze_graphs

from data_processing import process_dataset_mp, load_processed_graphs, save_processed_graphs, calculate_avg_superpixels
from models import SplineCNN

def debug_single_image_processing():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get the first image
    image, label = mnist_train[0]
    
    sp_params = {
        'wname': 'db1',
        'number': 0,
        'thresh_mult': 1,
        'verbose': True,  # Set to True for more output
        'multichannel': False
    }
    
    try:
        data_objs = process_image((image, label), sp_params)
        print("Successfully processed the image")
        print(f"Number of data objects created: {len(data_objs)}")
        for i, data_obj in enumerate(data_objs):
            print(f"Data object {i}:")
            print(f"  Number of nodes: {data_obj.num_nodes}")
            print(f"  Number of edges: {data_obj.num_edges}")
    except Exception as e:
        print(f"Error processing single image: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception traceback: {traceback.format_exc()}")
    
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
class ImageProcessingCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, image_hash):
        cache_file = os.path.join(self.cache_dir, f'{image_hash}.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, image_hash, data):
        cache_file = os.path.join(self.cache_dir, f'{image_hash}.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

def inspect_processed_data(graphs, num_samples=5):
    print(f"Total number of processed graphs: {len(graphs)}")
    for i in range(min(num_samples, len(graphs))):
        graph = graphs[i]
        if graph is not None:
            print(f"\nSample {i+1}:")
            print(f"  x shape: {graph.x.shape}")
            print(f"  edge_index shape: {graph.edge_index.shape}")
            print(f"  y shape: {graph.y.shape}")
            print(f"  Number of nodes: {graph.num_nodes}")
            print(f"  Number of edges: {graph.num_edges}")
        else:
            print(f"\nSample {i+1}: None (processing error)")

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def plot_training_progress(train_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'ro-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'go-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    torch.autograd.set_detect_anomaly(True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Train)", leave=False)
    for i, data in enumerate(pbar):
        try:
            batch_count += 1
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct/total:.4f}'})
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            print(f"Batch size: {data.num_graphs}")
            print(f"x shape: {data.x.shape}")
            print(f"edge_index shape: {data.edge_index.shape}")
            print(f"edge_attr shape: {data.edge_attr.shape}")
            continue
    
    if batch_count == 0:
        print("No batches were processed successfully.")
        return 0, 0
    return total_loss / batch_count, correct / total

def test(model, loader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} (Test) ", leave=False)
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs
            pbar.set_postfix({'Acc': f'{correct/total:.4f}'})
    return correct / total

def main(force_reprocess=False):
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

    sp_params = {
        'wname': 'db1',
        'level': None,
        'thresh_mult': 0.1,  # Reduce this value
        'verbose': False
    }

    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)

    if force_reprocess or not os.path.exists('train_graphs.pkl'):
        print("Processing training dataset...")
        train_graphs = process_dataset_mp(mnist_train, sp_params, batch_size=32, num_workers=8, cache_dir=cache_dir)
        save_processed_graphs(train_graphs, 'train_graphs.pkl')
    else:
        print("Loading preprocessed training data...")
        train_graphs = load_processed_graphs('train_graphs.pkl')

    if force_reprocess or not os.path.exists('test_graphs.pkl'):
        print("Processing test dataset...")
        test_graphs = process_dataset_mp(mnist_test, sp_params, batch_size=32, num_workers=8, cache_dir=cache_dir)
        save_processed_graphs(test_graphs, 'test_graphs.pkl')
    else:
        print("Loading preprocessed test data...")
        test_graphs = load_processed_graphs('test_graphs.pkl')
    
    avg_train_superpixels = calculate_avg_superpixels(train_graphs)
    avg_test_superpixels = calculate_avg_superpixels(test_graphs)
    print(f"Average number of superpixels in training set: {avg_train_superpixels:.2f}")
    print(f"Average number of superpixels in test set: {avg_test_superpixels:.2f}")

    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of test graphs: {len(test_graphs)}")

    try:
        print("Analyzing training graphs:")
        analyze_graphs(train_graphs)
    except Exception as e:
        print(f"Error analyzing training graphs: {str(e)}")
        traceback.print_exc()

    try:
        print("\nAnalyzing test graphs:")
        analyze_graphs(test_graphs)
    except Exception as e:
        print(f"Error analyzing test graphs: {str(e)}")
        traceback.print_exc()

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # In main.py, after creating train_loader
    for i, data in enumerate(train_loader):
        if i >= 5:  # Print info for first 5 batches
            break
        print(f"Batch {i}:")
        print(f"Number of graphs: {data.num_graphs}")
        print(f"x shape: {data.x.shape}")
        print(f"edge_index shape: {data.edge_index.shape}")
        print(f"edge_attr shape: {data.edge_attr.shape}")
        print(f"y shape: {data.y.shape}")
        print()

    model = SplineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)


    num_epochs = 75
    train_losses, train_accs, test_accs = [], [], []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        loss, train_acc = train(model, train_loader, optimizer, device, epoch)
        test_acc = test(model, test_loader, device, epoch)
        scheduler.step()
        
        #stores the loss and accuracy values for each epoch, allowing you to track the model's progress over time.
        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # After training, plot the results
    plot_training_progress(train_losses, train_accs, test_accs)

    print("Training completed.")

if __name__ == "__main__":
    main(force_reprocess=False)
    #Latest:
    #In this modified version, we've removed the pooling operation between the two SplineConv layers, as it was causing the dimension mismatch. 
    # If you want to keep the pooling, you'll need to adjust the number of input channels to the second SplineConv layer to match the output of the pooling layer.
    #def construct_graph(sp_image):
    #class SplineCNN(torch.nn.Module): - Put output in notes to claude


