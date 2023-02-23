# GDSSR
### GDSSR: Toward Real-World Ultra-High-Resolution Image Super-Resolution

>[IEEE Xplore](https://ieeexplore.ieee.org/document/10041757)<br>
> Yichen Chi, Wenming Yang, Yapeng Tian <br>
> Tsinghua University, University of Texas at Dallas


<p align="center">
  <img src="figs/im_c_real.PNG" width="50%"><img src="figs/im_c_sy.PNG" width="50%">
</p>

---

## 🔧 Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/chiyich/GDSSR.git
    cd GDSSR
    ```

2. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## Training (4 V100 GPUs)

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 gdssr/train.py -opt options/GDSSR_L1_x4.yml --launcher pytorch --auto_resume

python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 gdssr/train.py -opt options/GDSSR_GAN_x4.yml --launcher pytorch --auto_resume
```

## :european_castle: Model Zoo

Please download checkpoints from [Github Release](waiting).


## Testing

```bash
python inference_gdssr.py -opt (path to .yml file) --im_path (path to LR images) --model_path (path to checkpoint) --res_path (path to save SR images)

python Metric/LPIPS.py --folder_gt (path to HR images) --folder_restored (path to SR images)

python Metric/NIQE.py --folder_restored (path to SR images)

python Metric/DISTS.py --folder_gt (path to HR images) --folder_restored (path to SR images)
```

Results are in the `results` folder


## BibTeX

    @article{chi2023gdssr,
      title={GDSSR: Toward Real-World Ultra-High-Resolution Image Super-Resolution},
      author={Chi, Yichen and Yang, Wenming and Tian, Yapeng},
      journal={IEEE Signal Processing Letters},
      year={2023},
      publisher={IEEE}
    }

## 📧 Contact

If you have any question, please email `chiyich@yeah.net`.

## 🤗 Acknowledgement

Thanks to the following open-source projects:
- [MM-RealSR](https://github.com/TencentARC/MM-RealSR).
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- [CResMD](https://github.com/hejingwenhejingwen/CResMD).
- [CUGAN](https://github.com/HaomingCai/CUGAN).
