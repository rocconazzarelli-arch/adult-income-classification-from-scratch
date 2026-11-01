# Adult Income Classification from Scratch

A comparison of two supervised-learning approaches on the UCI Adult dataset:

- a fully connected neural network implemented with JAX and trained with several first-order optimizers;
- a scikit-learn k-nearest neighbors pipeline with preprocessing and cross-validated model selection.

## Results from the saved runs

| Model | Main result |
| --- | ---: |
| Neural network — RMSProp | 85.38% test accuracy |
| Neural network — AdaGrad | 85.15% test accuracy |
| Neural network — SGD with momentum | 84.37% test accuracy |
| k-NN | 84.37% test accuracy |
| k-NN | 0.893 ROC-AUC |

The k-NN classification report shows weaker recall for the minority `>50K` class. This is an important limitation of accuracy alone and motivates class-aware evaluation.

## What is implemented

### Neural network

- mixed numerical and categorical preprocessing;
- train/validation/test separation;
- dense architecture and loss functions implemented with JAX;
- manual training loops;
- RMSProp, AdaGrad, and SGD with momentum;
- hyperparameter search and learning-curve comparison.

### k-nearest neighbors

- `ColumnTransformer` preprocessing pipeline;
- scaling and one-hot encoding;
- stratified cross-validation;
- grid search over neighbors, distance metrics, and weighting;
- confusion matrix, ROC, precision–recall, and learning curves.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The notebooks download the public UCI Adult dataset when a local CSV is not provided.

## Repository structure

```text
.
├── notebooks/
│   ├── 01_ann_from_scratch.ipynb
│   └── 02_knn_pipeline.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Author

Rocco Nazzarelli

M.Sc. Computer Science and Engineering – Artificial Intelligence, Politecnico di Milano
