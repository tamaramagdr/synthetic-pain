# synthetic-pain

## About

Part of experiments for our paper [_The Expressive Power of Path based Graph Neural Networks_ (ICML 2024)](https://openreview.net/forum?id=io1XSRtcO8).

Please find the main part of the implementation [here](https://github.com/ocatias/ExpressivePathGNNs).

## Usage

Tested with Python 3.11.3.

To install requirements:
```pip install -r requirements.txt```

Run PAIN for EXP with path length 5:
```python -m src.synthetic.iso -d EXP -l 5```

Run PAIN for SR with path length 4 and marking neighbors:
```python -m src.synthetic.iso -d SR -l 4 -m```

If you use an IDE, make sure that the working directory is the root of the project.

## Citation

If you use our code please cite us as

```
@inproceedings{pathGNNs2024,
title={The Expressive Power of Path based Graph Neural Networks},
author={Drucks,Tamara and Graziani, Caterina and Jogl, Fabian and Bianchini, Monica and  Scarselli, Franco and Gärtner, Thomas },
booktitle={ICML},
year={2024},
url={https://openreview.net/forum?id=io1XSRtcO8}
}
```


## Misc

Feel free to tell me if you find any bugs :)
