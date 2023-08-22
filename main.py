import argparse

import numpy as np
import torch
from dataset import SumoTrafficDataset, MetrLALoader

from models import UNIN, InterpolationLSTM, GRIN, GaussianLSTM
from training import Trainer
import statistics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SUMO')  # SUMO or MetrLA
    parser.add_argument('--past_horizon', type=int, default=20)
    parser.add_argument('--predict_in', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_percent', type=float, default=0.5)
    parser.add_argument('--use_static', action='store_false')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--model', type=str, default='FUN-N')  # FUN-N, GRIN, GaussianLSTM, InterpolationLSTM

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    do_time_split = True
    mask_input = True
    eval_remain = True
    full_optimize = not mask_input
    torch.manual_seed(args.seed)
    models = {}
    if args.dataset == 'SUMO':
        hidden = 8
        # train_percent = 0.5
        base_percent = args.train_percent + (1 - args.train_percent) / 2
        if eval_remain:
            val_percent = args.train_percent
            test_percent = val_percent / 2
        else:
            val_percent = 0.5
            test_percent = 0.25
        label_count = 13 if args.use_static else 0
        eval_every = 100
        dataset = SumoTrafficDataset(reload=False, seed=args.seed, base_percent=base_percent,
                                     train_percent=args.train_percent,
                                     val_percent=val_percent, test_percent=test_percent,
                                     past_horizon=args.past_horizon, fully_observed=full_optimize,
                                     predict_in=args.predict_in, do_time_split=do_time_split)
    elif args.dataset == 'MetrLA':
        hidden = 32
        base_percent = args.train_percent + (1 - args.train_percent) / 2
        val_percent = 0.5
        test_percent = 0.25
        label_count = 0
        eval_every = 1000
        dataset = MetrLALoader(base_percent=base_percent, train_percent=args.train_percent, val_percent=val_percent,
                               test_percent=test_percent, past_horizon=args.past_horizon, predict_in=args.predict_in,
                               seed=args.seed, do_time_split=do_time_split)
    full_data = dataset.get_test().cuda()
    print(
        f"Evaluating  (L:{args.use_static}) for {args.dataset}, in {args.predict_in}, with {args.train_percent}, (seed:{args.seed}, Mask input:{mask_input})")
    data = dataset[0].to(device)

    steps = 100000
    break_count = 2
    learn_rate = 1e-3
    weight_decay = 0
    dropout = 0.25

    if args.model == 'FUN-N':
        model = UNIN(hidden=hidden, device=device, features=data.num_features, dropout=dropout,
                     labels=label_count)
    elif args.model == 'GRIN':
        model = GRIN(hidden=hidden, device=device, features=data.num_features, dropout=dropout,
                     labels=label_count)
    elif args.model == 'GaussianLSTM':
        model = GaussianLSTM(hidden=hidden, device=device, features=data.num_features, dropout=dropout,
                             labels=label_count)
    elif args.model == 'InterpolationLSTM':
        model = InterpolationLSTM(hidden=hidden, device=device, features=data.num_features, dropout=dropout,
                                  labels=label_count)
    model = model.to(device)
    trainer = Trainer(model, device, args.seed)
    trainer.train(dataset, steps=steps, eval_every=eval_every, break_count=break_count, learn_rate=learn_rate,
                  weight_decay=weight_decay, verbose=args.verbose, mask_input=mask_input, full_optimize=full_optimize)  #

    models[f'{args.model}{"+L" if args.use_static else ""}'] = model

    statistics.evaluate(train_data=dataset.get_train().cuda(), test_data=full_data, models=models, mean=True,
                        mean_timestep=False, batch_count=1, gaussian=True,
                        mask_input=mask_input)
