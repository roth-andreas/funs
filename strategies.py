import sklearn.metrics
import torch


def predict_mean(data, base_mask, val_mask, batch_count=1):

    total_mse = 0
    for b in range(batch_count):

        x_base = data.x[:, torch.reshape(base_mask, (-1,))]
        x_val = data.x[:, torch.reshape(val_mask, (-1,))]

        timestep_mean = torch.sum(x_base, 1) / len(x_base[0])
        mse = torch.zeros_like(timestep_mean)

        for t in range(len(mse)):
            for n in range(len(x_val[t])):
                mse[t] += torch.square(x_val[t][n] - timestep_mean[t])

            mse[t] = mse[t] / len(x_val[t])
        total_mse += mse.detach() / batch_count

    return total_mse


def preedict_global_mean(train_data, test_data, base_mask):
    mean = torch.mean(train_data.x[:, base_mask], dim=(0, 1))
    pred = torch.ones_like(test_data.x) * mean

    return pred


def predict_gaussian(data, base_mask, val_mask, sigma=1):
    kernel = gaussian_kernel(data, base_mask, val_mask, sigma)
    return gaussian(kernel, base_mask, val_mask, data)


def gaussian_kernel(data, base_mask, val_mask, sigma):
    kernel = rbf_kernel(data, sigma=sigma)
    return mask_kernel(kernel, base_mask, val_mask, sigma)


def mask_kernel(kernel, base_mask, val_mask, sigma):
    k = kernel[base_mask][:, base_mask]
    k_star = kernel[val_mask][:, base_mask]
    full_kernel = torch.matmul(k_star, torch.inverse(k + sigma ** 2 * torch.eye(k.shape[0]).cuda()))
    return full_kernel


def gaussian(full_kernel, base_mask, val_mask, data):
    pred = torch.zeros_like(data.x).to(data.x.device)

    pred[:, val_mask] = torch.matmul(full_kernel, data.x[:, base_mask])

    return pred


def rbf_kernel(data, sigma):
    # sk_dists = sklearn.metrics.pairwise_distances(data.coords.cpu().numpy())
    m = torch.max(data.coords)
    squared_distance = torch.cdist(data.coords / m, data.coords / m) ** 2

    gamma = 1 / (2 * sigma * sigma)
    kernel = torch.exp(-gamma * squared_distance)

    return kernel
