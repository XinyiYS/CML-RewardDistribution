# Collaborative Generative Modelling
The code for the paper "Incentivizing Collaboration in Machine Learning via Synthetic Data Rewards".

## Requirements
1. Linux machine (experiments were run on Ubuntu 18.04.5 LTS and Ubuntu 20.04.2 LTS)
2. NVIDIA GPU with CUDA 11.0 (experiments were run with NVIDIA GeForce GTX 1080 Ti and NVIDIA TITAN V)
3. Anaconda (alternatively, you may install the packages in `environment.yml` manually)

## Setup
In the main `CGM` directory,
1. Run the following script to download the datasets.
```shell
bash download.sh
```
2. Run the following command to install the required Python packages into a new environment named CGM using Anaconda.
```shell
conda env create -f environment.yml
```

## Running experiments
In the main `CGM` directory,
1. Change current environment to the CGM environment.
```shell
conda activate CGM
```
2. Run the desired experiment. Valid values for `dataset` are `{creditratings, creditcard, mnist, cifar}`, valid values for `split` are `{equaldisjoint, unequal}`, and valid values for `inv_temp` are any non-negative real number.
```shell
python cgm.py with ${dataset} split=${split} inv_temp=${inv_temp}
```
```
# Example: to run the experiment on the creditcard dataset with the equal disjoint split and inv_temp = 1
python cgm.py with creditcard split=equaldisjoint inv_temp=1
```
3. To replicate the correlation metrics in the paper for any dataset and split, run the following Python script to compute and display the metrics after all experiments with `inv_temp = {1, 2, 4, 8}` have completed.
```
python metrics.py wih ${dataset}
```
4. To replicate the downstream supervised learning experiments in the paper for any dataset and split, run the following Python script to compute and display the metrics after all experiments with `inv_temp = {1, 2, 4, 8}` have completed.
```
python supervised.py wih ${dataset}
```

## License
This code is released under the MIT License.
