# CIFAR Baselines

This project implements existing baseline models on CIFAR dataset.
Try to report the best performance for each baseline, although it involves cumbersome tuning of hyperparameters.

## CIFAR-10

### WRN28x1-BN-ReLU

| Method  | # Ensemble | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-      | :-:        | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD     | 1          | 0.9231 | 0.2801 | 0.0396 | 0.1031 | 0.0000 | 0.2346 | 0.0043 | 0.2476 | 1.6781 |
| DE-2    | 2          | 0.9423 | 0.1997 | 0.0141 | 0.1375 | 0.2619 | 0.1917 | 0.0160 | 0.1978 | 1.3086 |
| DE-4    | 4          | 0.9460 | 0.1701 | 0.0086 | 0.1560 | 0.2528 | 0.1697 | 0.0119 | 0.1769 | 1.1031 |
| DE-8    | 8          | 0.9507 | 0.1543 | 0.0118 | 0.1649 | 0.2440 | 0.1543 | 0.0119 | 0.1653 | 1.0016 |

## CIFAR-100

### WRN28x1-BN-ReLU

| Method  | # Ensemble | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-      | :-:        | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD     | 1          | 0.6917 | 1.2164 | 0.1124 | 0.6259 | 0.0000 | 1.1002 | 0.0125 | 1.1395 | 1.5102 |
| DE-2    | 2          | 
| DE-4    | 4          | 
| DE-8    | 8          | 
