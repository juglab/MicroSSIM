# MicroSSIM

[![License](https://img.shields.io/pypi/l/microssim.svg?color=green)](https://github.com/juglab/MicroSSIM/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/microssim.svg?color=green)](https://pypi.org/project/microssim)
[![Python Version](https://img.shields.io/pypi/pyversions/microssim.svg?color=green)](https://python.org)
[![CI](https://github.com/juglab/MicroSSIM/actions/workflows/ci.yml/badge.svg)](https://github.com/juglab/MicroSSIM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/juglab/MicroSSIM/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/MicroSSIM)



MicroSSIM is an image measure aimed at addressing the shortcomings of the Structural
Similarity Index Measure (SSIM), in particular in the context of microscopy images. Indeed,
in microscopy, degraded images (e.g. lower signal to noise ratio) often have a different
dynamic range than the original images. This can lead to a poor performance of SSIM.

The measure normalizes the images using background subtraction and a more appropriate 
range estimation. It then estimates a scaling factor used to scale the image
to the target (original image or ground truth). The metric is then computed
similarly to the SSIM. 

MicroSSIM is easily extensible to other SSIM-like measures, such as Multi-Scale SSIM 
(MS-SSIM), for which we provide an example.

See the [paper](https://arxiv.org/abs/2408.08747) for more details.

## Installation

```bash
pip install microssim
```


## Usage

```python
import numpy as np
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity

rng = np.random.default_rng(42)
N = 5
gt = 200 + rng.integers(0, 65535, (N, 256, 256)) # stack of different images
pred = rng.poisson(gt) / 10

# using the convenience function
result = micro_structural_similarity(gt, pred)
print(f"MicroSSIM: {result} (convenience function)")

# using the class allows fitting a large dataset, then scoring a subset
microssim = MicroSSIM()
microssim.fit(gt, pred) # fit the parameters

for i in range(N):
    score = microssim.score(gt[i], pred[i]) # score a single pair
    print(f"MicroSSIM ({i}): {score}")

# compare with SSIM from skimage
for i in range(N):
    score = structural_similarity(gt[i], pred[i], data_range=65535)
    print(f"SSIM ({i}): {score}")
```

The code is similar for MicroMS3IM.

## Tips for deep learning

MicroSSIM was developed in the context of deep-learning, in which SSIM is often used
as a measure to compare denoised and ground-truth images. The tips presented here are
valid beyond deep-learning.

The larger the dataset, the better the estimate of the scaling factor will be. Therefore,
it is recommended to fit the measure on the entire dataset (e.g. the whole training 
dataset). Once the data fitted, the `MSSIM` class has registered the parameters used
for normalization and scaling. You can then score a subset of the data (e.g. the validation
or test datasets) using the `score` method.



## Cite us

If you use MicroSSIM in your research, please cite us:

Ashesh, Ashesh, Joran Deschamps, and Florian Jug. "MicroSSIM: Improved Structural Similarity for Comparing Microscopy Data." arXiv preprint arXiv:2408.08747 (2024). [link](https://arxiv.org/abs/2408.08747).
