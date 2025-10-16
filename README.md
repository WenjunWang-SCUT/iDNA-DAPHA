# iDNA_DAPHA

This repository contains the model code, domain-adaptive pretrained weights (dual-scale and four-scale), and fine-tuned weights on all benchmark datasets for "iDNA-DAPHA: a generic framework for methylation prediction via domain-adaptive pretraining and hierarchical attention".

## Models

The following models (weights) are provided for fine-tuning and evaluation:

- Domain-adaptive pretrained models for fine-tuning

| Variant | Description | Download |
| :--- | :--- | :--- |
| `Dual-Scale` (3- & 6-mers) | The primary model in our work, balancing performance and efficiency. | [Link](https://drive.google.com/file/d/1t3Db41Pti4jxjXojkqcIKI3bjOGPyiu3/view?usp=sharing) |
| `Four-Scale` (3- to 6-mers) | For users prioritizing maximal performance over computational cost. | [Link](https://drive.google.com/file/d/1wlQjloip0VxOJGXDrG4X5JX4ZA31F94C/view?usp=sharing) |

- Fine-tuned models for evaluation

| Dataset | Download | Dataset | Download |
| :--- | :--- | :--- | :--- |
| `4mC_C.equisetifolia` | [Link](https://drive.google.com/file/d/1xJ-irPvdyhSqunvApOWB_bOCC99tjQ72/view?usp=sharing) | `4mC_F.vesca` | [Link](https://drive.google.com/file/d/1A99nrNSsM85L-2MrKX3VkOfbuMeOHmXY/view?usp=sharing) |
| `4mC_S.cerevisiae` | [Link](https://drive.google.com/file/d/1IaXbcflo3aoEx7B8E8Cjhvy99jXO-O6C/view?usp=sharing) | `4mC_Tolypocladium` | [Link](https://drive.google.com/file/d/1NQRC2pgwfw6MkaYyXpDuPeGJk4MoJxxM/view?usp=sharing) |
| `5hmC_H.sapiens` | [Link](https://drive.google.com/file/d/1e4dTsq9zTm3F8hr4mEErlmqw6gPQ27AU/view?usp=sharing) | `5hmC_M.musculus` | [Link](https://drive.google.com/file/d/18pF0T6YCuiVS2INg76FICQh6Uj5Hc7gi/view?usp=sharing) |
| `6mA_A.thaliana` | [Link](https://drive.google.com/file/d/1AABQF9VNFL3nTRWlCkq6lm7CgBrCa24o/view?usp=sharing) | `6mA_C.elegans` | [Link](https://drive.google.com/file/d/1th5TwzMUioXpTlBdOqD9Cw952vbwx2lO/view?usp=sharing) |
| `6mA_C.equisetifolia` | [Link](https://drive.google.com/file/d/1tnGpgVE33FHlhR_qSaZUAEjsyMvgHZBy/view?usp=sharing) | `6mA_D.melanogaster` | [Link](https://drive.google.com/file/d/1U_-Aok99m7zsGjafGZfz6fsO7Tr8h32u/view?usp=sharing) |
| `6mA_F.vesca` | [Link](https://drive.google.com/file/d/1wsxdQ_AddhzATQSmfQtFktyMGWzW14I_/view?usp=sharing) | `6mA_H.sapiens` | [Link](https://drive.google.com/file/d/1uiLY2xMJ1Orhim4Pv2L9flZDufWlqOFU/view?usp=sharing) |
| `6mA_R.chinensis` | [Link](https://drive.google.com/file/d/1hGXcteM_Fu-kDy-zsIRFmF20dZciLKEM/view?usp=sharing) | `6mA_S.cerevisiae` | [Link](https://drive.google.com/file/d/1fkD-VJzJ5P22y_AwbxmoKfROb3HLvq7l/view?usp=sharing) |
| `6mA_Tolypocladium` | [Link](https://drive.google.com/file/d/1yaSFjMAYFaCeL0UxGQl9fddxq00RsdZP/view?usp=sharing) | `6mA_T.thermophile` | [Link](https://drive.google.com/file/d/1VTKgT6ALv-_RCXcitQvXKGerU8y2-VFO/view?usp=sharing) |
| `6mA_Xoc BLS256` | [Link](https://drive.google.com/file/d/1TPUCOlZsNBfwVszKLAbO8DNB5nJdbVh-/view?usp=sharing) |

## Basic Directory

- You can change model structure for fine-tuning in `model/FusionDNAbert.py`, or for pretraining in `model/FusionDNAbert_dap.py`.

- You can modify training process and dataset processing for fine-tuning in `frame/ModelManager.py` and `frame/DataManager.py`, or for pretraining in `frame/ModelManager_dap.py` and `frame/DataManager_dap.py`.

- The model can be fine-tuned or evaluated with pretrained or trained weights using `main/finetune_and_eval.py`, and pretrained using `main/pretrain_dap.py`.

- You can modify parameters in `configuration/config_init.py` to fine-tune or pretrain models.

- The used benchmark datasets are included in `data/DNA_MS`.

Note: 
Before running domain-adaptive pretraining, fine-tuning, or evaluation, please ensure the [`DNABERT`](https://github.com/jerryji1993/DNABERT) pretrained model (including model weights, tokenizer, and vocab.txt) is downloaded and placed in the 'pretrain/' folder. It is necessary for proper parameter initialization and consistent tokenization. If you want to use the four-scale strategy, modify the value of config.kmers in both the fine-tuning script (`main/finetune_and_eval.py`) and the pretraining script (`main/pretrain_dap.py`).

## Run Commands
```bash
# Enter the `main/` folder
cd main

# Fine-tune the model using the domain-adaptive pretraining weights 
python finetune_and_eval.py -path-params <PATH_TO_DOMAIN-ADAPTIVE_PRETRAINING_WEIGHTS> -dataset <DATASET_NAME>

# Evaluate the model performance using the fine-tuned weights
python finetune_and_eval.py -do-eval -path-params <PATH_TO_FINETUNED_WEIGHTS> -dataset <DATASET_NAME>

# Pretrain the model with domain-adaptive pretraining
python pretrain_dap.py

Note: The --dataset argument accepts values such as 4mC_C.equisetifolia, 4mC_F.vesca, 5hmC_H.sapiens, and other benchmark dataset names.
``` 

## Contact

For any questions or issues related to iDNA_DAPHA, please create an issue — We will do our best to help.
