# Forecasting Unobserved Node States with spatio-temporal Graph Neural Networks

This repository contains an implementation of the research paper you can find [here](https://ieeexplore.ieee.org/abstract/document/10031042?casa_token=0Sz9YcuTzmsAAAAA:3ckRA8DjTerGuw_nHo5GpCBCCgY9IKdwuLhKmpC3R0TdNbPGIWyGkHWhv1-nGmDRZFKHFED0). The implementation aims to reproduce the results and findings described in the paper for traffic prediction on two datasets: SUMO and MetrLA.

## Prerequisites

Before using this implementation, make sure you have the following requirements satisfied:

- Python (>=3.9)
- PyTorch (>=1.8.0)
- Pytorch Geometric (>=1.7.0)

## Data

The SUMO data can be downloaded [here](https://www.dropbox.com/scl/fo/etcsipbottsio52i65173/h?rlkey=vv5aqp3srvh63vsrjv45wksve&dl=0). Both files should be placed in the data/raw_dir folder.

The MetrLA data will be downloaded automatically when running the main.py script for the first time.

## Usage

Execute the main.py script with the following arguments:

* --dataset: Choose the dataset to use for traffic prediction. Options are 'SUMO' or 'MetrLA'.
* --past_horizon: Set the number of past time steps to consider for prediction.
* --predict_in: Set the time step in the future to predict. Use 0 for immediate prediction.
* --seed: Set the random seed for reproducibility.
* --train_percent: Specify the percentage of data to use for training.
* --use_static: Add this flag to enable using static features (if available).
* --verbose: Add this flag to enable verbose output.
* --model: Choose the prediction model. Options are 'FUN-N', 'GRIN', 'GaussianLSTM', 'InterpolationLSTM'.

Here's an example command:

```console
python main.py --dataset SUMO --past_horizon 20 --predict_in 0 --seed 0 --train_percent 0.5 --use_static --verbose --model FUN-N
```

Replace the values of the arguments as needed to match the experiment settings from the research paper.


## Citation

If you found this implementation helpful in your research, please consider citing our paper.

```bibtex
@inproceedings{roth2022forecasting,
  title={Forecasting Unobserved Node States with spatio-temporal Graph Neural Networks},
  author={Roth, Andreas and Liebig, Thomas},
  booktitle={2022 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={740--747},
  year={2022},
  organization={IEEE}
}
```