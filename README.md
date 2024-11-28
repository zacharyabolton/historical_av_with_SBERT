# Historical Authorship Verification with SBERT

This project utilizes Siamese SBERT models to perform historical authorship verification.

## Setup Development Environment

Follow these steps to prepare the development environment for this project:

### 1. Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone <repository-url>
```

Navigate to the cloned repository directory:

```bash
cd historical_av_with_SBERT
```

### 2. Create and Activate the Virtual Environment

Create a virtual environment using Python 3.9:

```bash
python3.9 -m venv venv
```

Activate the virtual environment:

- On Unix or MacOS:

  ```bash
  source venv/bin/activate
  ```

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

### 3. Install Required Packages

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Set Up IPython Kernel

Install a project-specific IPython kernel to use within Jupyter Lab:

```bash
python -m ipykernel install --user --name="historical_av_with_SBERT-env" --display-name="Historical AV with SBERT Kernel"
```

### 5. Launch Jupyter Lab

Start Jupyter Lab:

```bash
jupyter lab
```

Select the "Historical AV with SBERT Kernel" from the kernel options to work within the project environment.

## Maintenance Commands

### Update `requirements.txt`

If you install any new packages during development, update the `requirements.txt` file to reflect these changes:

```bash
pip freeze > requirements.txt
```

## Additional Information

Ensure that you have Python 3.9 installed on your system to use with this project as it is required for compatibility with the specific version of PyTorch used in the project.

---

Feel free to report issues or contribute to this project by submitting pull requests or contacting the maintainer.
