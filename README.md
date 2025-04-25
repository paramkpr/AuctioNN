# AuctioNN

AuctioNN is a mechanism-design project that explores how **neural networks** can be leveraged to outperform traditional auctions (like second-price auctions) in the context of online advertising. Our goal is to allocate ad impressions among multiple advertisers to maximize overall value (in terms of conversions, revenue, fairness, etc.).

We use a real-world ad impression dataset (provided by [Claritas](https://www.claritas.com/)) with features such as device information, geolocation data, time since last impression, and conversion outcomes. By learning complex relationships in this data, our model aims to deliver *better* allocation decisions compared to traditional rule-based systems.

---
## Usage
AuctioNN provides a command line interface for data wrangling, model training, and allocation simulation. You can view the available commands by running:
```bash
python main.py --help
```

### Preprocessing
The `preprocess` command cleans and preprocesses the raw dataset and saves the cleaned data to a parquet file.
```bash
python main.py preprocess --impressions-file data/raw/impressions.parquet \
    --conversions-file data/raw/conversions.parquet \
    --output-file data/processed/clean_data.parquet
```

### Fitting Preprocessors
The `fit-preprocessors` command fits the preprocessors (encoders, scalers) on the training split of the cleaned data and saves them to a directory.
```bash
python main.py fit-preprocessors --cleaned-data-file data/processed/clean_data.parquet \
    --output-dir ./preprocessors
```

### Training (WIP)
The `train` command trains the neural network model on the cleaned data using the preprocessors.
```bash
python main.py train --data-path data/processed/clean_data.parquet \
    --preprocessor-dir ./preprocessors
```

### Allocating (WIP)
The `allocate` command runs the allocation mechanism using the trained model.
```bash
python main.py allocate --model-path ./models/neural_net.pth \
    --data-path data/processed/clean_data.parquet \
    --output-dir ./allocations
```

### Evaluating (WIP)
The `evaluate` command evaluates the performance of the allocation mechanism.
```bash
python main.py evaluate --data-path data/processed/clean_data.parquet \
    --output-dir ./evaluations
```

---

## Directory Structure
The following is a typical directory structure for the AuctioNN project:
```
├── docs                      - LaTeX setup for the project write up
│   └── README.md
├── main.py                   - Main CLI entry point for the project
├── notebooks                 - Jupyter notebooks for exploratory data analysis and model development
│   └── README.md
├── requirements.txt          - List of dependencies
├── setup.py                  - Setup script for the project (allows statements like `import auctionn.models.neural_net` anywhere in the project)
├── src
│   ├── __init__.py           
│   ├── data_processing       - Code for processing the dataset
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── mechanism             - Code for the mechanism design (e.g. second-price auction)
│   │   ├── __init__.py
│   │   └── allocation.py
│   └── models                - Code for the neural network models
│       ├── __init__.py
│       └── neural_net.py
├── web_app                   - Code for the web app (optional: if time allows)
│   └── README.md
├── .gitignore
└── README.md
```
## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/paramkpr/AuctioNN.git
```

1. Setup a virtual environment and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. Run the main script:
```bash
python main.py
python main.py --help # to see the available options
```

3. **IMPORTANT** If you want to run the code in the notebooks, you jupyter kernel needs to use this virtual environment so that it's the same for all users.
You need to install the kernel in the venv by running the following command:
```bash
python -m ipykernel install --user --name auctionn --display-name "Python (auctionn)"
```
Now when you run `jupyter notebook`, you should see the kernel (or the option to select the kernel) as `Python (auctionn)`.
If you're using VSCode, you can select the kernel by going to the `Python: Select Interpreter` option in the command palette.

## Contributing
Before making any changes, please create a new branch:
```bash
git checkout -b feature/new-feature
```

Make your changes and commit them:
```bash
git add .
git commit -m "Add new feature"
```

Push your changes to the branch:
```bash
git push origin feature/new-feature
```

Create a pull request on GitHub. Of course, you can self-review your changes. However, making PRs for any changes is encouraged because then we'll all be on the same page.


## Documentation and Style
1. **Docstrings**: Please use PEP 257 style docstrings for all functions and classes.
2. **Type hints**: Please use type hints for all functions and classes. (e.g. `def function_name(param1: int, param2: str) -> bool:`)
3. **Code readability**: Please follow the [PEP 8](https://pep8.org/) style guide for Python code.
4. **Comments**: Please add comments to the code to explain "why" behind the code in more complex functions.


## Getting the dataset from Claritas
You'll need to set up AWS CLI on your machine and create a profile with access to the Claritas bucket. Credentials are stored in the repository secrets. 

1. Install AWS CLI:
```bash
brew install awscli
```

2. Create a profile:
```bash
aws configure
```
3. Commmands you'll want:
```bash
aws --profile auctionn s3 ls s3://calpoly-artsai --human-readable --summarize --recursive
aws --profile auctionn s3 ls s3://calpoly-artsai/prediction_allocation_logic/ --human-readable --summarize --recursive
aws --profile auctionn s3 cp s3://calpoly-artsai/prediction_allocation_logic/ ./data --recursive
```


## Auction Model

We have an online model where we have a stream of impressions from an ad exchange. 
Our model represents a marketing company that has a group of advertisers and places bids on their behalf on the ad exchange.

We have a stream of impressions and a stream of conversions.
