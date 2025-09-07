## Prequisites
===========
- **Python version**:  `3.11.9`
- **Pytorch version** : `2.4.1+cu118`


## Setup Instructions
Create an conda environment and install all the required packages with the following command:
```bash
conda create -n myenv python=3.11.9
conda activate myenv
```

Then, install the required packages:
```bash
pip install -r requirements.txt
```
## Running the Script
To train and test: run the following command:
```bash

python train.py --config config/config.yaml
```

## Preparing Your Own Dataset

Download the [JetClass dataset](https://zenodo.org/records/6619768/files/JetClass_Pythia_test_20M.tar?download=1) 
### Steps
1. Get the `JetClass_Pythia_test_20M.tar` file from the link above and extract it.

2.  Split the dataset into three parts:
   - **Training:** 2M samples  
   - **Validation:** 1M samples  
   - **Testing:** 1M samples  

3. Use the provided notebook [`data/dataset_h5.ipynb`](data/dataset_h5.ipynb) to:
   - Load the raw dataset  
   - Preprocess it  
   - Save it in `.h5` format (required for training)

---

After this step, you’ll have ready-to-use `.h5` files for training, validation, and testing.
