## Create environment
All required package except for pytorch can be install using following command:
```bash
conda env create -f env.yml -n [new_env_name]
```
then you need to install pytorch independently:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Data Preprocessing

This section describes the scripts used for data preprocessing and their purposes:

1. **mid_preprocess.ipynb**
   - **Purpose**: Generates domain-specific CSV files from `ml_raw_data`.
   - **Usage**: Open and run this notebook in Jupyter to output the CSV files.

2. **data_preprocess.py**
   - **Purpose**: Preprocesses the domain CSV files into a format suitable for model input.
   - **Usage**: Run the script from the command line:
     ```bash
     python data_preprocess.py [dataset_name]
     ```
        where `[dataset_name]` can be "Movie_lens_main", "RQ4_dataset", "RQ5_dataset"
        
3. **calculate_IR.ipynb**
    - **Purpose**: Calculate group interaction ratio (GIR) for **Group-aware Contrastive Learning**.
   - **Usage**: Set the variable `main_dataset` as the name of the current dataset to generate GIR for the specific dataset.



## Item Generator Pre-training

Follow the steps below to pre-train the item generator model. The pre-trained model will be stored in the `generator_model` folder.

### Requirements
Ensure you have the following directory structure before running the pre-training script:
- **data_dir**: The CDSR scenario folder (e.g., `action_comedy`).
- **dataset**: The parent directory of `data_dir`.

### Running the Pre-training Script

Execute the following command from your terminal, substituting `[data_dir]` and `[dataset]` with the actual paths:

```
python run_generator.py --data_dir [data_dir] --dataset [dataset]
```


## Model Training


### Training the Recommendation Model

To train the model and obtain the recommendation results, execute the command below:

```
python train_rec.py
```
### Retrieving Results for Specific Research Questions

For detailed results associated with different research questions, follow the guidelines provided for each RQ:
* **For RQ1, RQ2, RQ4, and RQ5:**
Run the following command, replacing `[num]` with the research question number (e.g., 1, 2, 4, or 5) and `[mode]` with the experiment mode. This command will execute the appropriate experiment script for the specified research question and mode.
 **General command format:**
    ```
    python RQ[num]/[mode]/run.py
    ```
    **Example:**
    ```
    python RQ5/substitute_mode/run.py
    ```
   
* **For RQ3:**
 To obtain results for RQ3, execute the command below. This script performs parameter tuning to optimize the model settings:
    ```
    python param_tune_result/param_tuning.py
    ```
