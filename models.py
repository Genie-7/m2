import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add
from torch_geometric.utils import coalesce

class WavePool(torch.nn.Module):
    def __init__(self):
        super(WavePool, self).__init__()

    def forward(self, x, edge_index, edge_attr, batch):
        device = x.device
        
        # Compute the number of nodes for each graph in the batch
        num_nodes = scatter_add(torch.ones_like(batch), batch, dim=0)
        
        # Compute the maximum number of nodes in any graph in the batch
        max_num_nodes = num_nodes.max().item()
        
        # Convert node features to dense batch
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)
        
        # Perform 2x2 average pooling on each graph separately
        x_pooled = F.avg_pool2d(x_dense.view(x_dense.size(0), 1, max_num_nodes, -1), 2).squeeze(1)
        
        # Flatten the pooled features and remove padding
        x_pooled = x_pooled.view(-1, x_pooled.size(-1))[mask.view(-1, max_num_nodes)[:, ::2].view(-1)]
        
        # Update edge_index
        row, col = edge_index
        row_pooled = row // 2
        col_pooled = col // 2
        edge_index_pooled = torch.stack([row_pooled, col_pooled], dim=0)
        
        # Remove duplicate edges
        edge_index_pooled, edge_attr = coalesce(edge_index_pooled, edge_attr, num_nodes=x_pooled.size(0))
        
        # Update batch
        batch_pooled = batch[::2]
        
        return x_pooled, edge_index_pooled, edge_attr, batch_pooled

class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=10, num_spline_bases=5):
        super(SplineCNN, self).__init__()
        self.conv1 = SplineConv(in_channels, hidden_channels, dim=2, kernel_size=num_spline_bases)
        self.conv2 = SplineConv(hidden_channels, 2*hidden_channels, dim=2, kernel_size=num_spline_bases)
        self.fc1 = torch.nn.Linear(2*hidden_channels, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)