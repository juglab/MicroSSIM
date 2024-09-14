# MicroSSIM

MicroSSIM is an image metrics aimed at addressing the shortcominbs of the Structural
Similarity Index (SSIM), in particular in the context of microscopy images. Indeed,
in microscopy, degraded images (e.g. lower signal to noise ratio) often have a different
dynamic range than the original images. This can lead to a poor performance of SSIM.

The metrics normalizes the images using background subtraction and a more appropriate 
range estimation. It then estimates a range-invariant factor used to scale the image
compared to the target (original image or ground truth). The metric is then computed
similarly to the SSIM. 

MicroSSIM is easily extensible to other SSIM-like metrics, such as Multi-Scale SSIM 
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

# using the class
microssim = MicroSSIM()
microssim.fit(gt, pred) # fit the parameters

for i in range(N):
    score = microssim.score(gt[i], pred[i])
    print(f"MicroSSIM ({i}): {score}")

# compare with SSIM
for i in range(N):
    score = structural_similarity(gt[i], pred[i], data_range=65535)
    print(f"SSIM ({i}): {score}")
```

## Cite us

Ashesh, Ashesh, Joran Deschamps, and Florian Jug. "MicroSSIM: Improved Structural Similarity for Comparing Microscopy Data." arXiv preprint arXiv:2408.08747 (2024). [link](https://arxiv.org/abs/2408.08747).