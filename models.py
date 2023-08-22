import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch_geometric as pyg
import strategies


class MeanPassing(MessagePassing):
    def __init__(self):
        super(MeanPassing, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


class GRIN(nn.Module):
    def __init__(self, hidden=16, features=2, device='cpu',
                 bias=True, interval=10, dropout=0, labels=0, **kwargs):
        super(GRIN, self).__init__()
        self.reset_mpnn = GATv2Conv(features + labels + hidden + 1, hidden, add_self_loops=False)
        self.update_mpnn = GATv2Conv(features + labels + hidden + 1, hidden, add_self_loops=False)
        self.cell_mpnn = GATv2Conv(features + labels + hidden + 1, hidden, add_self_loops=False)
        self.final_mpnn = GATv2Conv(features + labels + hidden + 1, hidden, add_self_loops=False)
        self.linear1 = nn.Linear(hidden, features)
        self.linear2 = nn.Linear(hidden * 2, features)
        self.pred_linear = nn.Linear(hidden * 2, features)
        self.hidden = hidden
        self.use_labels = labels > 0

    def forward(self, x, edge_index, mask, labels, edge_weight):
        num_timesteps, num_nodes, num_features = x.size()
        mask = mask.view(-1, 1)

        pred = torch.ones_like(x[0]).cuda()
        hidden = torch.ones(num_nodes, self.hidden).cuda()
        masked_input = torch.zeros(num_timesteps, num_nodes, num_features).cuda()
        masked_input = torch.where(mask.view(1, -1, 1), x, masked_input)
        x_2 = masked_input[0]  # torch.ones_like(x[0]).cuda()

        preds = torch.zeros(num_timesteps, num_nodes, num_features).cuda()

        for t in range(num_timesteps):
            feature_tensor = torch.cat((x_2, mask), dim=1)
            if labels is not None:
                feature_tensor = torch.cat((feature_tensor, labels), dim=1)
            reset = torch.sigmoid(self.reset_mpnn(torch.cat((feature_tensor, hidden), dim=1), edge_index, edge_weight))
            update = torch.sigmoid(
                self.update_mpnn(torch.cat((feature_tensor, hidden), dim=1), edge_index, edge_weight))
            cell = torch.tanh(
                self.cell_mpnn(torch.cat((feature_tensor, reset * hidden), dim=1), edge_index, edge_weight))
            hidden = update * hidden + (1 - update) * cell
            hidden = torch.nn.functional.dropout(hidden, p=0.25, training=self.training)

            y_1 = self.linear1(hidden)
            x_1 = torch.where(mask, masked_input[t], y_1)

            feature_tensor = torch.cat((x_1, mask, hidden), dim=1)
            if labels is not None:
                feature_tensor = torch.cat((feature_tensor, labels), dim=1)
            s = self.final_mpnn(feature_tensor, edge_index, edge_weight)
            s = torch.relu(s)
            s = torch.nn.functional.dropout(s, p=0.25, training=self.training)

            y_2 = self.linear2(torch.cat((s, hidden), dim=1))
            x_2 = torch.where(mask, masked_input[t], y_2)
            masked_input[t] = x_2

        return masked_input


class UNIN(nn.Module):
    def __init__(self, hidden=16, features=2, device='cpu',
                 bias=True, interval=10, dropout=0, labels=0, **kwargs):
        super(UNIN, self).__init__()
        self.static_linear = nn.Sequential(
            nn.Linear(2 + labels, hidden))
        mpnn_size = features + hidden * 2
        self.reset_mpnn = GATv2Conv(mpnn_size, hidden, add_self_loops=True)
        self.update_mpnn = GATv2Conv(mpnn_size, hidden, add_self_loops=True)
        self.cell_mpnn = GATv2Conv(mpnn_size, hidden, add_self_loops=True)
        self.final_mpnn = GATv2Conv(features + hidden * 2, hidden, add_self_loops=True)
        self.linear1 = nn.Linear(hidden, features)
        self.linear2 = nn.Linear(hidden * 2, features)
        self.pred_linear = nn.Sequential(
            nn.Linear(hidden * 2, features)
        )
        self.hidden = hidden
        self.fill_mpnn = GATv2Conv(features + hidden * 2, features, add_self_loops=True)
        self.deep_mpnn = GATv2Conv(features + hidden * 2, hidden, add_self_loops=True)
        self.use_labels = labels > 0

    def forward(self, x, edge_index, mask, labels, edge_weight, data):
        num_timesteps, num_nodes, num_features = x.size()
        mask = mask.view(-1, 1)

        hidden = torch.zeros(num_nodes, self.hidden).cuda()
        masked_input = torch.zeros_like(x).cuda()
        masked_input = torch.where(mask.view(1, -1, 1), x, masked_input)

        preds = torch.zeros(num_timesteps, num_nodes, num_features).cuda()

        feature_tensor = torch.cat((mask, ~mask), dim=1).float()
        if labels is not None:
            feature_tensor = torch.cat((feature_tensor, labels), dim=1)
        static = torch.relu(self.static_linear(feature_tensor))

        for t in range(num_timesteps):
            y_2 = self.fill_mpnn(torch.cat((masked_input[t], static, hidden), dim=1), edge_index,
                                 edge_weight)
            x_2 = torch.where(mask, masked_input[t], y_2)

            feature_tensor = torch.cat((x_2, static), dim=1)
            reset = torch.sigmoid(self.reset_mpnn(torch.cat((feature_tensor, hidden), dim=1), edge_index, edge_weight))
            update = torch.sigmoid(
                self.update_mpnn(torch.cat((feature_tensor, hidden), dim=1), edge_index, edge_weight))
            cell = torch.tanh(
                self.cell_mpnn(torch.cat((feature_tensor, reset * hidden), dim=1), edge_index, edge_weight))
            hidden = update * hidden + (1 - update) * cell
            hidden = torch.nn.functional.dropout(hidden, p=0.25, training=self.training)

            feature_tensor = torch.cat((x_2, static, hidden), dim=1)
            s = self.final_mpnn(feature_tensor, edge_index, edge_weight)
            s = torch.relu(s)
            s = torch.nn.functional.dropout(s, p=0.25, training=self.training)

            pred = self.pred_linear(torch.cat((s, static), dim=1))
            preds[t] = pred

        return preds


class Interpolation(nn.Module):
    def __init__(self):
        super(Interpolation, self).__init__()
        self.model = MeanPassing()

    def forward(self, x, edge_index):
        with torch.no_grad():
            i = 0
            while (x == 0).any() and i < 20:
                A = pyg.utils.to_dense_adj(edge_index)[0]
                A[torch.sum((x != 0), dim=(0, 2)) == 0, :] = 0
                filtered_edge_index = pyg.utils.dense_to_sparse(A)[0]
                # filtered_edge_index = torch.stack(
                #    [edge for edge in edge_index.T if (x[:, edge[0]] != 0).any()]).T
                output = self.model(x, filtered_edge_index)
                x = torch.where(x == 0, output, x)
                i += 1
        return x


class InterpolationLSTM(nn.Module):
    def __init__(self, hidden=16, features=2, device='cpu',
                 bias=True, interval=10, dropout=0, labels=0):
        super(InterpolationLSTM, self).__init__()
        self.interpolation = Interpolation()
        self.rnn = nn.LSTM(features, features)
        self.hidden = hidden
        self.use_labels = False

    def forward(self, x, edge_index, mask, labels, edge_weight, data):
        masked_input = torch.zeros_like(x).cuda()
        masked_input = torch.where(mask.view(1, -1, 1), x, masked_input)

        interp_x = self.interpolation(masked_input, edge_index)
        output, _ = self.rnn(interp_x)
        return output


class GaussianLSTM(nn.Module):
    def __init__(self, hidden=16, features=2, device='cpu',
                 bias=True, interval=10, dropout=0, labels=0):
        super(GaussianLSTM, self).__init__()
        self.rnn = nn.LSTM(features, features)
        self.use_labels = False
        self.kernel = None

    def forward(self, x, edge_index, mask, labels, edge_weight, data):
        if self.kernel is None:
            self.kernel = strategies.rbf_kernel(data, 3)
        masked_input = torch.zeros_like(x).cuda()
        masked_input = torch.where(mask.view(1, -1, 1), x, masked_input)

        masked_kernel = strategies.mask_kernel(self.kernel, mask, ~mask, 3)
        gaussian_pred = strategies.gaussian(masked_kernel, mask, ~mask, data)

        masked_input = torch.where(mask.view(1, -1, 1), masked_input, gaussian_pred)

        output, _ = self.rnn(masked_input)
        return output
