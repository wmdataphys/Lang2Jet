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
## Running the Pretrained Model
To test using a pretrained model: run the following command:
```bash

python train.py --config config/config.yaml
```