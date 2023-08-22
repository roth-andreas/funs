import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import copy


class Trainer:
    def __init__(self, model, device='cpu', seed=1):
        self.model = model
        self.device = device
        torch.manual_seed(seed)

    def train(self, dataset, steps=100, eval_every=10, break_count=5, weight_decay=0,
              learn_rate=0.005, verbose=True, mask_input=True, full_optimize=False):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        not_improved_for = 0
        optimal_loss = np.inf

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        data_iter = iter(dataloader)
        losses = []
        start = time.time()
        for i in range(1, steps):
            self.model.train()
            loss = 0

            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            # for data in dataloader:
            if full_optimize:
                loss_masks = ['train_mask', 'base_mask', 'val_mask', 'test_mask']
            else:
                loss_masks = ['train_mask', 'base_mask']
            _, loss_list = forward(self.model, data, None, loss_masks, reduction='none', mask_input=mask_input)
            loss += torch.mean(torch.cat(loss_list, dim=1))
            # + self.calc_loss(data.x[:, data.train_mask], out[:, data.train_mask], 'MAE')
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % eval_every == 0):
                mean_loss = np.mean(losses)
                losses = []
                self.model.eval()
                _, (val_loss, test_loss) = eval(self.model, [dataset.get_val()],
                                                input_masks=['base_mask', 'train_mask'],
                                                eval_masks=['val_mask', 'test_mask'], short=False,
                                                mask_input=mask_input)
                if (val_loss < optimal_loss):
                    optimal_loss = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    not_improved_for = 0
                else:
                    not_improved_for += 1

                if verbose:
                    print(
                        f"Step: {i} train loss: {mean_loss:.4f} val loss: {val_loss:.4f}, test error: {test_loss:.4f} ({time.time() - start:.4f}s)")

                if (not_improved_for >= break_count):
                    break
                start = time.time()


        self.model.load_state_dict(best_model)
        _, (val_loss, test_loss) = eval(self.model, [dataset.get_test()], input_masks=['base_mask', 'train_mask'],
                                        eval_masks=['val_mask', 'test_mask'],
                                        mask_input=mask_input)
        _, (train_loss,) = eval(self.model, [dataset.get_test()], input_masks=['base_mask'], eval_masks=['train_mask'],
                                mask_input=mask_input)

        print(f"Optimum: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}")


def eval(model, dataloader, input_masks, eval_masks, reduction='mean', short=False, mask_input=False):
    model.eval()

    losses = [[] for _ in eval_masks]
    outputs = []
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            data = data.cuda()
            if len(input_masks) == 1:
                mask = data[input_masks[0]]
            else:
                mask = torch.add(*[data[mask] for mask in input_masks])

            out, loss_list = forward(model, data, mask, eval_masks, reduction='none', short=short,
                                     mask_input=mask_input)

            outputs.append(out)
            for i, eval_mask in enumerate(eval_masks):
                losses[i].append(loss_list[i].cpu())
            if step > 1000:
                break

    return outputs, [reduce_loss(loss, reduction) for loss in losses]


def reduce_loss(losses, reduction='mean'):
    torch_losses = torch.cat(losses)
    if reduction == 'mean':
        return torch.mean(torch_losses)
    elif reduction == 'nodes':
        return torch.mean(torch_losses, dim=1)
    return torch_losses


def calc_loss(targets, predictions, reduction='mean'):
    return F.mse_loss(predictions, targets, reduction=reduction)


def forward(model, data, mask=None, eval_masks=['train_mask'], reduction='mean', short=False, mask_input=True):
    data = data.cuda()
    # setup labels
    if model.use_labels:
        labels = data.attributes.to(data.x.device)
    else:
        labels = None
    if 'edge_weight' in data:
        edge_weight = data.edge_weight
    else:
        edge_weight = None

    if short:
        # p = random.randint(0, len(data.x)-500)
        data.x = data.x[:10000]
        data.y = data.y[:10000]

    if mask_input:
        input_mask = mask if mask is not None else data.base_mask
    else:
        input_mask = torch.ones(data.x.shape[1], dtype=torch.bool).cuda()
    preds = model(data.x, data.edge_index, input_mask, labels, edge_weight, data)
    # loss = calc_loss(data.x[:, data[val_mask]], out[:, data[val_mask]], 'MSE')
    if mask is None:
        preds = preds  # [-1:]
        target = data.y  # [-1:]
    else:
        target = data.y
    losses = []
    for val_mask in eval_masks:
        losses.append(calc_loss(target[:, data[val_mask]], preds[:, data[val_mask]], reduction=reduction))

    return preds, losses
