# synthetic-pain

## About

Part of experiments for our paper [https://openreview.net/forum?id=io1XSRtcO8](_The Expressive Power of Path based Graph Neural Networks_ (ICML 2024)).

Please find the main part of the implementation [](here).

## Usage

Tested with Python 3.11.3.

To install requirements:
```pip install -r requirements.txt```

For EXP with path length 5:
```python -m src.synthetic.iso -d EXP -l 5```

For SR with path length 4 and marking neighbors:
```python -m src.synthetic.iso -d SR -l 4 -m```

If you use an IDE, make sure that the working directory is the root of the project.

## Misc

Feel free to tell me if you find any bugs :)
