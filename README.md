# Learning Domain-Robust Bioacoustic Representations for Mosquito Species Classification with Contrastive Learning and Distribution Alignment

[Source paper ICASSP 2026](https://ieeexplore.ieee.org/abstract/document/11464393)

## Join the BioDCASE 2026 Challenge
This repository may also be useful for participants interested in the **BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC) Challenge**, jointly organised by the **University of Oxford, King’s College London, and the University of Surrey**.

The challenge focuses on a key real-world question:  
**Can mosquito species classifiers still work when recordings come from new locations, devices, and acoustic environments?**

We warmly welcome participants to join the challenge and help advance robust mosquito monitoring under real recording conditions.

- Challenge website: https://biodcase.github.io/challenge2026/task5
- Baseline code: https://github.com/Yuanbo2020/CD-MSC
- Challenge Dataset: https://zenodo.org/records/19095788

# Citation
If you find this work useful, please consider citing our paper as
```bibtex
@INPROCEEDINGS{11464393,
  author={Hou, Yuanbo and Liu, Zhaoyi and Shen, Xin and Roberts, Stephen},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Learning Domain-Robust Bioacoustic Representations for Mosquito Species Classification with Contrastive Learning and Distribution Alignment}, 
  year={2026},
  volume={},
  number={},
  pages={15207-15211},
  doi={10.1109/ICASSP55912.2026.11464393}}
```

## Dataset

1. Please download the dataset from here (https://zenodo.org/records/17287037).

2. Unzip it and move it to the `Dataset` folder under `Code/application`.

3. Make sure this file exists before running the scripts:
   `Code/application/Dataset/Mosquito_sound_8species_4domains.pickle`.

## Training

Run training from the `Code/application` directory:

```bash
cd application
python Training.py
```

Optional arguments:

- `-lr_rate` (default: `0.0005`)
- `-cs_temperature` (default: `0.01`)
- `-batch` (default: `64`)
- `-epochs` (default: `500`)

Example:

```bash
python Training.py -lr_rate 0.0005 -cs_temperature 0.01 -batch 64 -epochs 500
```

During training, a new experiment folder (e.g., `sy_lr...`) is created in `Code/application`, including:

- `logs/` for training and validation logs
- `model/` for checkpoints

## Evaluation

Run evaluation from the `Code/application` directory:

```bash
cd application
python Evaluation.py
```

By default, evaluation loads:
`Code/application/Pretrained_models/DR_BioL_CNN.pth`.

To evaluate `DAT_CNN`, edit `Evaluation.py` and set:

```python
model_name = 'DAT_CNN'
```

Outputs are saved to `Code/application/Inference`, including:

- confusion matrix figures (`*_cm.png`, `*_cm_normalize.png`)
- ROC figures and micro-AUC text files (`*_ROC_AUC_*.pdf`, `*_roc_micro_*.txt`)
- PR figures and micro-AP text files (`*_MSC_PR_*.pdf`, `*_pr_*.txt`)



