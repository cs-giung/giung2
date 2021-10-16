# Diversity Matters When Learning From Ensembles

This repository is the official implementation of [Diversity Matters When Learning From Ensembles]() (NeurIPS 2021).

## Training

To train DE-4 teachers with WRN28x10 on CIFAR-100, run the following commands:
```
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_0/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_1/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_2/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_3/
```

To train BE-4 students with WRN28x10 on CIFAR-100, run the following commands:
```
python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 4.0 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KD_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 4.0 --kd-method-name gaussian \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDGaussian_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 4.0 --kd-method-name ods_l2 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDODS_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 4.0 --kd-method-name c_ods_l2 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDConfODS_0/
```

## Evaluation

## Citation

If you find this useful in your research, please consider citing our paper:
```
@inproceedings{nam2021diversity,
  title     = {Diversity Matters When Learning From Ensembles},
  author    = {Giung Nam and Jongmin Yoon and Yoonho Lee and Juho Lee},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}
```
