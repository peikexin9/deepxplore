# DeepXplore: Systematic DNN testing  (SOSP'17)
See the SOSP'17 paper [DeepXplore: Automated Whitebox Testing of Deep Learning Systems](http://www.cs.columbia.edu/~suman/docs/deepxplore.pdf) for more details.
## Prerequisite
### Python
The code should be run using python 2.7.12, Tensorflow 1.3.0, Keras 2.0.8, PIL, h5py, and opencv-python

### Tensorflow
```bash
sudo pip install tensorflow
```
if you have gpu,
```bash
pip install tensorflow-gpu
```

### Keras
```bash
pip install keras
```
To set Keras backend to be tensorflow (two options):
```bash
1. Modify ~/.keras/keras.json by setting "backend": "tensorflow"
2. KERAS_BACKEND=tensorflow python gen_diff.py
```

### PIL
```bash
pip install Pillow
```

### h5py
```bash
pip install h5py
```

### opencv-python
```bash
pip install opencv-python
```

### Mimicus
Install from [here](https://github.com/srndic/mimicus).

## File structure
+ **MNIST** - MNIST dataset.
+ **ImageNet** - ImageNet dataset.
+ **Driving** - Udacity self-driving car dataset.
+ **PDF** - Benign/malicious PDFs captured from VirusTotal/Contagio/Google provided by Mimicus.
+ **Drebin** - Drebin Android malware dataset.

# To run
In every directory
```bash
python gen_diff.py
```

# Note
The trained weights are provided in each directory (if required).
Drebin's weights are not part of this repo as they are too large to be hosted on GitHub. Download from [here](https://drive.google.com/drive/folders/0B4otJeEcboCaQzFpYkJwb2h3WG8?resourcekey=0-ns2toseJWe6qVS0nOl6rnw&usp=sharing) and put them in ./Drebin/.

Note that as DeepXplore use randomness for its exploration, you should fix the seed of the random number generator if you want deterministic and reproducable results. An example is shown below.   
```python
import numpy as np
import random

random.seed(1)
np.random.seed(1)
```

# Coming soon
How to test your own DNN models.
