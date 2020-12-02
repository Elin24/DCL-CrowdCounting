# DCL-CrowdCounting

This is an official implementaion of the paper "Density-aware Curriculum Learning for Crowd Counting", completed in November 2019, accepted by T-CYB in October 2020.

[IEEE link](https://ieeexplore.ieee.org/document/9275392)[pdf download](http://crabwq.github.io/pdf/2020%20Density-aware%20Curriculum%20Learning%20for%20Crowd%20Counting.pdf)

![DCL-Crowd Counting](images/dclflow.png)

This repository shows how PSCC is trained with/without DCL strategy. Relevant experiment processes are shown in `process_reports`.

- `normal.log` demonstrates the process of PSCC under random sampling.
- `curriculum.log` demonstrates the process of PSCC under density-aware curriculum learning.
- `*.txt` shows the configration and verification results during training.

# Requirements

- Python 2.7 (It is 2019 when submiting the paper. **py3** will be supported in the future.)
- Pytorch 1.2.0
- TensorboardX
- torchvision 0.4.0
- easydict

# Dara preparation

1. Download the original ShanghaiTech Dataset [link: [Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) / [BaiduPan](https://pan.baidu.com/s/1nuAYslz)]
2. generate the density maps using the `datasets/generate_data.py` (using **Python 3** because of the *f-string*) according to the README in datasets.
3. modify the `dataset/SHHA/setting.py` th specify the path of dataset.

# Training

1. modify the training parameters in `config.py`.
    - Without DCL, set `__C.DCL_CONF['work'] = False`
    - With DCL, set `__C.DCL_CONF['work'] = True`
2. `python train.py`

# Experiment Results

| PSCC |MAE | MSE |
|:-:|:-:|:-:|
Random Sampling | 66.82 | 109.35
Density-aware CL | **64.97** | **107.96**

# Citation

If you use the code, please cite the following paper:

```
@ARTICLE{9275392,
  author={Q. {Wang} and W. {Lin} and J. {Gao} and X. {Li}},
  journal={IEEE Transactions on Cybernetics}, 
  title={Density-Aware Curriculum Learning for Crowd Counting}, 
  year={2020},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TCYB.2020.3033428}}
```
