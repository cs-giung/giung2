# CIFAR Baselines

This project implements existing baseline models on CIFAR dataset.
Try to report the best performance for each baseline, although it involves cumbersome tuning of hyperparameters.

## Standard Baselines

### WRN28x1-BN-ReLU on CIFAR-10

> All models are trained on the first 45,000 examples of the train split of CIFAR-10; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.9231 | 0.2801 | 0.0396 | 0.1031 | -      | 0.2346 | 0.0043 | 0.2476 | 1.6781 |
| DE-2       | 2     | 0.9423 | 0.1997 | 0.0141 | 0.1375 | 0.2619 | 0.1917 | 0.0160 | 0.1978 | 1.3086 |
| DE-4       | 4     | 0.9460 | 0.1701 | 0.0086 | 0.1560 | 0.2528 | 0.1697 | 0.0119 | 0.1769 | 1.1031 |
| DE-8       | 8     | 0.9507 | 0.1543 | 0.0118 | 0.1649 | 0.2440 | 0.1543 | 0.0119 | 0.1653 | 1.0016 |
| Dropout    | 1     | 0.9356 | 0.2529 | 0.0341 | 0.0845 | -      | 0.2086 | 0.0061 | 0.2099 | 1.6383 |
| MC-Dropout | 30    | 0.9361 | 0.2215 | 0.0226 | 0.1127 | 0.0539 | 0.2012 | 0.0055 | 0.2012 | 1.4227 |
| DUQ        | 1     | 0.9284 | 0.2950 | 0.0359 | 0.1031 | 0.0000 | 0.2535 | 0.0082 | 0.2265 | 1.4297 |
| DUQ-GP     | 1     | 

### WRN28x1-BN-ReLU on CIFAR-100

> All models are trained on the first 45,000 examples of the train split of CIFAR-100; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.6917 | 1.2164 | 0.1124 | 0.6259 | -      | 1.1002 | 0.0125 | 1.1395 | 1.5102 |
| DE-2       | 2     | 0.7247 | 1.0003 | 0.0266 | 0.7747 | 1.0543 | 0.9797 | 0.0318 | 0.9920 | 1.2102 |
| DE-4       | 4     | 0.7499 | 0.8928 | 0.0279 | 0.8834 | 1.0527 | 0.8928 | 0.0316 | 0.9007 | 1.0156 |
| DE-8       | 8     | 0.7663 | 0.8357 | 0.0528 | 0.9454 | 1.0395 | 0.8301 | 0.0261 | 0.8326 | 0.9039 |

### WRN28x10-BN-ReLU on CIFAR-100

> All models are trained on the first 45,000 examples of the train split of CIFAR-100; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.8050 | 0.8007 | 0.0584 | 0.6338 | -      | 0.7922 | 0.0398 | 0.8383 | 1.1453 |

## Baselines for Bayesian Interpretation

### PR20-FRN-SiLU

> All models are trained on the first 40,960 examples of the train split of CIFAR-10; the last 9,040 examples of the train split are used as the validation split. For a clear Bayesian interpretation of the inference procedure, (1) we do not use any data augmentation, and (2) batch normalization is replaced with filter response normalization.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.8519 | 0.5415 | 0.0802 | 0.2041 | -      | 0.4632 | 0.0199 | 0.4559 | 1.5414 |
| DE-2       | 2     | 0.8716 | 0.4242 | 0.0313 | 0.2780 | 0.4483 | 0.4053 | 0.0199 | 0.3939 | 1.2617 |
| DE-4       | 4     | 0.8873 | 0.3536 | 0.0178 | 0.3195 | 0.4576 | 0.3524 | 0.0194 | 0.3539 | 1.0773 |
