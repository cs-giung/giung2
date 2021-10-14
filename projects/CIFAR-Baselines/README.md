# CIFAR Baselines

This project implements existing baseline models on CIFAR dataset.
Try to report the best performance for each baseline, although it involves cumbersome tuning of hyperparameters.

## CIFAR-10

### WRN28x1-BN-ReLU

| Method  | # Ensemble | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-      | :-:        | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD     | 1          | 0.9231 | 0.2801 | 0.0396 | 0.1031 | 0.0000 | 0.2346 | 0.0043 | 0.2476 | 1.6781 |

## CIFAR-100

### WRN28x1-BN-ReLU

| Method  | # Ensemble | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-      | :-:        | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD     | 1          | 0.6917 | 1.2164 | 0.1124 | 0.6259 | 0.0000 | 1.1002 | 0.0125 | 1.1395 | 1.5102 |
