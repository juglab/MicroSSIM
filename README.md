# Objective
This is the official implementation of [MicroSSIM](https://arxiv.org/abs/2408.08747), accepted at BIC, ECCV 2024. 
## Installation
We will soon release the package on PyPI. For now, you can install the package by cloning the repository and running the following command:
```bash
git clone git@github.com:juglab/MicroSSIM.git
cd MicroSSIM
pip install -e .
```
## Usage
```python
from microssim import MicroSSIM, MicroMS3IM
gt: N x H x W
pred: N x H x W

ssim = MicroSSIM() # or MicroMS3IM()
ssim.fit(gt, pred)

for i in range(N):
    score = ssim.score(gt[i], pred[i])
    print('SSIM score for', i, 'th image:', score)

```