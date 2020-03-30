# Volcanet

Radar image based cnn classifier for volcanoes.

## Installation
Make sure you have a **virtual** python environment, with **python vesion 3.7**.

Then just install the package and it's dependancies.
```bash
pip install -e .
```

## Usage
### Make inferences
Make predictions from png images in a directory.
```bash
python volcanet/app.py [/path/to/test_file.wav]
```
You will get a list of predictions and their index.

### Retraining from scratch
If your would like to retrain the classifier:
```bash
python volcanet/training/train.py
```

### Evaluate the model
```bash
python volcanet/training/experiments.py
```
