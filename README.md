# Historical Authorship Verification with SBERT

This project utilizes Siamese SBERT models to perform historical authorship verification.

## Setup Development Environment

Follow these steps to prepare the development environment for this project:

### 1. Clone the Repository
- **Start by cloning this repository to your local machine**:
```bash
git clone https://github.com/zacharyabolton/historical_av_with_SBERT.git
```

- **Navigate to the cloned repository directory**;
```bash
cd historical_av_with_SBERT
```
### 2. Environment Setup
- **Create and activate virtual environment (Python 3)**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # Unix/MacOS
  venv\Scripts\activate     # Windows
  ```

- **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
### 4. Set Up IPython Kernel (optional)

- **Install a project-specific IPython kernel to use within Jupyter Lab**:
```bash
python -m ipykernel install --user --name="kernel_name" --display-name="Kernel Name"
```

- **Start Jupyter Lab**:
```bash
jupyter lab
```

Select the kernel from the kernel options to work within the project environment.

---
## How to Apply This Framework

Researchers looking to apply this framework to new authorship verification problems can follow these steps:

### 1. Dataset Preparation
- **Organize text files** in a directory structure following the LILA pattern:
  ```
  /dataset_root/
  ├── cleaned/
  │   ├── A/        # Known author documents
  │   ├── notA/     # Imposter documents
  │   └── U/        # Documents for inference (optional)
  └── metadata.csv  # Document metadata
  ```
- **Create metadata.csv** with these columns:
  - `file` (str): Filename including extension
  - `author_short` (str): Short alphanumeric, whitespace-stripped author name identifier
  - `author` (str): Full author name
  - `genre` (str): Text genre for balancing
  - `imposter_for` (str): Target class the imposter represents
  - `canonical_class_label` (str): 'A', 'notA', or 'U'
  - `class` (int): 1 for class A, 0 for notA, blank for U
  - `omit` (bool): Flag to exclude document from dataset
  - `num_words` (int): Document word count

- **Apply text normalization** using the provided script:
  ```bash
  python scripts/text_normalizer.py /path/to/dataset_root "A,notA,U"
  ```

- **Generate distorted views** (recommended) using the provided script:
  ```bash
  python scripts/text_distorter.py /path/to/dataset_root "A,notA,U" "300,3000,20000"
  ```

### 2. Configure Hyperparameters
Based on experimental findings, start with these recommended values:
- **Margin parameters**: `margin_s=0.75`, `margin_d=0.25` for modified contrastive loss
- **Batch size**: 32 is optimal for performance but can be reduced if memory-constrained
- **Gradient accumulation**: Increase `accumulation_steps` (default: 1) to effectively increase batch size without additional memory
- **Chunk size**: 512 tokens (maximum for all-MiniLM-L12-v2)
- **Learning rate**: 0.00002 with linear decay
- **Gradient clipping**: `max_norm=1.0` to stabilize training
- **Training duration**: 
  - 3 epochs are often sufficient for single-problem datasets
  - 30-200 epochs recommended for multi-problem datasets

### 3. Training Execution
- **Run training** with command:
  ```bash
  python src/main.py "experiment_name" "/path/to/dataset/normalized" 32 1 512 0.75 0.25 0.000001 20720 5 3 0.00002 -m 1.0
  ```
- **Monitor training** through CLI output and logged metrics.
- **Results location**: All outputs are saved to `/path/to/dataset_root/model_out/experiment_name/`, including:
  - Loss plots (per fold and summary)
  - Confusion matrices
  - Validation metrics
  - Saved models for inference

### 4. Model Evaluation
- **K-fold validation** metrics are automatically calculated.
- **Decision thresholds**: Start with recommended p1=0.45 and p2=0.54
- **Examine confusion matrices** to identify potential class biases.
- **Analyze the impact of distortion** by comparing performance across views.

### 5. Inference on Unknown Texts
- **Automatic pairing**: The `LILADataset` class handles inference pairing automatically by setting `letters=True`
- **Each unknown chunk** is paired with 10 known-author chunks by default.
- **Analyzing inference results**: Review the distribution of similarity scores to determine authorship likelihood
- **Apply decision thresholds**: Use the same thresholds established during validation.

### 6. Performance Considerations
- **Memory constraints**: Use gradient accumulation by increasing the `accumulation_steps` parameter
- **Runtime optimization**: Reduce `num_pairs` for faster training on more minor problems
- **GPU utilization**: The code automatically uses CUDA if available, with fallback to MPS or CPU

---
## Maintenance Commands

**Update `requirements.txt`**

If you install any new packages during development, update the  `requirements.txt` file to reflect these changes:

```bash
pip freeze > requirements.txt
```

Feel free to report issues or contribute to this project by submitting pull requests or contacting the maintainer.