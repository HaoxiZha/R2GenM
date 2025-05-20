# R2Gen

R2Gen implements the model described in ["Generating Radiology Reports via Memory-driven Transformer"](https://arxiv.org/pdf/2010.16056.pdf) (EMNLP 2020). It provides tools to train the model on common radiology datasets and reproduce the results from the paper.

## Requirements
- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`

## Pretrained Models
Links to pretrained checkpoints for IU X-Ray and MIMIC-CXR can be found in [data/r2gen.md](data/r2gen.md).

## Datasets
This repository supports two datasets:

- **IU X-Ray** – download the data from [Google Drive](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and place the files under `data/iu_xray/`.
- **MIMIC-CXR** – download the data from [Google Drive](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and place the files under `data/mimic_cxr/`.

## Training
Use the provided scripts to train R2Gen:

```bash
bash run_iu_xray.sh    # Train on IU X-Ray
bash run_mimic_cxr.sh  # Train on MIMIC-CXR
```

Each script calls `main.py` with the recommended parameters. Feel free to modify them to experiment with different settings.

## Citation
If you use this code, please cite:

```text
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020"
}
```
