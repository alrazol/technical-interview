# Technical Interview: Submission

This package can be used to develop and evaluate ML models to solve power grid related problems. It uses Grid2Op as an underlying data emulator.

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies with [uv](https://github.com/astral-sh/uv)

This project uses **uv** for ultra-fast dependency management and virtual environments. Installation details are provided through the link.

```bash
# 1. Create and sync the environment
uv sync --all-groups

# 2. Activate the environment
source .venv/bin/activate
```

### 2️⃣ Populate the `.env` file

If not already present, copy the sample file at the project root:

```bash
cp .env.sample .env
```

Populate the file, for example:

```
cache_dir=/Users/.../project/cache
artifacts_dir=/Users/.../project/artifacts
```

### 3️⃣ Run CLI

The entrypoint to the package is the `src/cli.py` file. You can run an experiment after having created an experiment configuration file. The package is shipped with the config at `src/configs/linear_regression.yaml` so you can directly use it by running:

```
python src/cli.py --experiment-name linear_regression --experiment-config-path src/configs/linear_regression.yaml
```

You can change the `--experiment-name` argument if you want to save under a different dir to compare across experiments.

⚠️ **Warning:** Make sure you activated you environment, if not, run `source .venv/bin/activate` before running the CLI.

### 4️⃣ Submission Notebook

The notebook for the data exploration question is `explore_data.ipynb`.
