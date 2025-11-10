# LLMKT

## Webscraper

### Run the following to set up a Python environment:

```bash
python -m venv webscraper
source webscraper/bin/activate
pip install -r requirements.txt
```

### Run post-installation setup

```bash
crawl4ai-setup
```

### Note: If you encounter any browser-related issues, you can install them manually:

```bash
python -m playwright install --with-deps chromium
```

### Verify your installation

```bash
crawl4ai-doctor
```

## Merge-Script

### Run the following to set up a Conda environment:

```bash
conda create -n merge_env python=3.11 -y
conda activate merge_env
```

### Run the following to set up requirements.txt:

```bash
pip install -r requirements.txt
```

### Run the merge script:

```bash
cd merge_script
# Copy datasets to this folder

python merge_all_datasets.py
```
