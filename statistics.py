import torch
import matplotlib.pyplot as plt
import numpy as np
import strategies
import training
from models import MeanPassing


def evaluate(train_data, test_data, models=None, mean=False, mean_timestep=False, gaussian=None, batch_count=1,
             mask_input=True):
    print("test performance:")
    mask = test_data.base_mask + test_data.train_mask  # + data.val_mask
    val_mask = test_data.val_mask
    test_mask = test_data.test_mask
    evaluate_all_models(train_data=train_data, test_data=test_data, observed_mask=mask, val_mask=val_mask, test_mask=test_mask,
                        mean=mean, mean_timestep=mean_timestep, gaussian=gaussian, models=models,
                        batch_count=batch_count, mask_input=mask_input)


def evaluate_all_models(train_data, test_data, observed_mask, val_mask, test_mask, models=None, mean=False, mean_timestep=False,
                        gaussian=None, batch_count=1, mask_input=True):
    val_mask = torch.reshape(val_mask, (-1,))
    test_mask = torch.reshape(test_mask, (-1,))

    losses = {}
    preds = {}
    groundtruth = test_data.y[:, test_mask]
    preds['Groundtruth'] = groundtruth
    print(f'Groundtruth test loss: {torch.nn.functional.mse_loss(groundtruth, groundtruth)}')

    interp_preds = predict_interpolation(test_data, observed_mask)
    interp_preds, interp_loss = preds_to_loss(interp_preds, groundtruth, test_mask, test_data, 'Interpolation')
    losses['Interpolation'] = interp_loss
    preds['Interpolation'] = interp_preds
    # interp_preds = torch.mean(interp_preds, dim=1)

    for name, model in models.items():
        mpnn_pred = predict_gnn(test_data, model, ['base_mask', 'train_mask'], ['test_mask'], mask_input=mask_input)
        mpnn_pred, mpnn_loss = preds_to_loss(mpnn_pred, groundtruth, test_mask, test_data, name)
        losses[name] = mpnn_loss
        preds[name] = mpnn_pred
    if (mean):
        mean_pred = strategies.preedict_global_mean(train_data, test_data, observed_mask)
        mean_pred, mean_loss = preds_to_loss(mean_pred, groundtruth, test_mask, test_data, 'Mean')
        losses['Mean'] = mean_loss
        preds['Mean'] = mean_pred
    if (mean_timestep):
        mean_pred_t = strategies.predict_mean(test_data, observed_mask, test_mask, batch_count=batch_count)
    if 'coords' in test_data and (gaussian != None):
        best_sigma = 3
        best_loss = np.inf
        for sigma in [1, 1.5, 3]:
            gaussian_pred = strategies.predict_gaussian(data=test_data, base_mask=observed_mask, val_mask=val_mask,
                                                        sigma=sigma)
            gaussian_pred = gaussian_pred[:, val_mask]
            gaussian_loss = torch.nn.functional.mse_loss(gaussian_pred, groundtruth, reduction='none')
            if torch.mean(gaussian_loss) < best_loss:
                best_loss = torch.mean(gaussian_loss)
                best_sigma = sigma
        gaussian_pred = strategies.predict_gaussian(data=test_data, base_mask=observed_mask, val_mask=test_mask,
                                                    sigma=best_sigma)
        gaussian_pred, gaussian_loss = preds_to_loss(gaussian_pred, groundtruth, test_mask, test_data, 'Gaussian')
        losses['Gaussian'] = gaussian_loss
        preds['Gaussian'] = gaussian_pred

    plot_MSE_time({name: torch.mean(loss, dim=1) for name, loss in losses.items()})


def preds_to_loss(preds, groundtruth, mask, data, prefix):
    preds = preds[:, mask]
    loss = torch.nn.functional.mse_loss(preds, groundtruth, reduction='none')
    loss = torch.mean(loss, dim=1)
    print(f"Test {prefix} loss: {torch.mean(loss)}")
    return preds, loss


def predict_interpolation(data, mask):
    with torch.no_grad():
        model = MeanPassing()
        preds = data.x * mask.view(1, -1, 1)
        i = 0
        while (preds == 0).any() and i < 20:
            edge_index = torch.stack(
                [edge for edge in data.edge_index.T if (preds[:, edge[0]] != 0).any()]).T
            output = model(preds, edge_index)
            preds = torch.where(preds == 0, output, preds)
            i += 1
    return preds


def predict_gnn(data, gnn, base_mask, val_mask, mask_input=True):
    output, (loss,) = training.eval(gnn, [data], base_mask, val_mask, reduction='none',
                                    mask_input=mask_input)
    output = torch.cat(output)
    return output


def plot_MSE_time(values):
    legend = []
    for name, value in values.items():
        value = value.cpu().detach().numpy()
        plt.plot(torch.linspace(0, len(value), len(value)), value)
        legend.append(f"{name}")

    plt.legend(legend)
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('MSE')
    plt.show()
