#!/usr/bin/env bash
set -e

echo "Scaffolding testing framework ..."

mkdir -p config
mkdir -p resources/models
mkdir -p artifacts/screenshots artifacts/payloads artifacts/plots
mkdir -p domains/api tests/api
mkdir -p domains/etl tests/etl
mkdir -p domains/ml tests/ml
mkdir -p tests/smoke
mkdir -p tests/workflows
mkdir -p workflows cli
mkdir -p framework
mkdir -p .github/workflows
mkdir -p src/framework
mkdir -p package


mkdir -p framework/{core,domains/{api,etl,ml,workflows},tests/{api,etl,ml,workflows},config,tools}
mkdir -p ci docker

# ========================================================
# ENV CONFIGS
# ========================================================
cat > config/config.dev.json <<'EOF'
{
  "env": "dev",
  "base_url": "https://api.dev.example.com",
  "tolerance": 0.05
}
EOF

cat > config/config.dev.json <<'EOF'
{
  "env": "dev",
  "base_url": "https://api.qa.example.com",
  "tolerance": 0.05
}
EOF

# ========================================================
# CORE UTILITIES
# ========================================================
cat > framework/core/config_loader.py <<'EOF'
import json
from pathlib import Path

def load_config(env: str):
    path = Path("config") / f"config.{env}.json"
    return json.loads(path.read_text())
EOF

# CLI
cat > framework/core/cli.py <<'EOF'
import argparse, subprocess

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["api","etl","ml","workflows","smoke"])
    parser.add_argument("--env", default="dev")
    args = parser.parse_args()

    if args.task == "smoke":
        subprocess.check_call(["pytest","-m","smoke"])
    else:
        subprocess.check_call(["pytest", f"tests/{args.task}", "-vv", f"--env={args.env}"])

if __name__ == "__main__":
    run()
EOF

# Allure utilities
cat > framework/core/allure_utils.py <<'EOF'
import allure
import matplotlib.pyplot as plt

def attach_json(name, obj):
    import json
    allure.attach(json.dumps(obj, indent=2), name=name,
                  attachment_type=allure.attachment_type.JSON)

def attach_text(name, text):
    allure.attach(text, name=name,
                  attachment_type=allure.attachment_type.TEXT)

def attach_plot(name, xs, ys):
    plt.figure()
    plt.plot(xs, ys)
    path = f"/tmp/{name}.png"
    plt.savefig(path)
    allure.attach.file(path, name=name,
                       attachment_type=allure.attachment_type.PNG)
EOF

# CLI wrapper
cat > framework/run_framework.sh <<'EOF'
#!/usr/bin/env bash
python -m framework.core.cli "$@"
EOF
chmod +x framework/run_framework.sh

# ========================================================
# DOMAINS
# ========================================================

# ---- API ----
cat > framework/domains/api/client.py <<'EOF'
import httpx, asyncio

async def get_ip(base):
    async with httpx.AsyncClient(base_url=base) as c:
        r = await c.get("/ip")
        r.raise_for_status()
        return r.json()
EOF

cat > tests/api/test_ip_schema.py <<'EOF'
import pytest, asyncio
from framework.core.config_loader import load_config
from framework.domains.api.client import get_ip
from framework.core.allure_utils import attach_json

@pytest.mark.api
def test_ip(request):
    cfg = load_config(request.config.getoption("--env"))
    data = asyncio.run(get_ip(cfg["base_url"]))
    attach_json("response", data)
    assert "origin" in data

@pytest.mark.parametrize("endpoint", ["ip","uuid"])
@pytest.mark.api
def test_generic_param(endpoint, request):
    cfg = load_config(request.config.getoption("--env"))
    data = asyncio.run(get_ip(cfg["base_url"]))
    assert isinstance(data, dict)
EOF

# ---- ETL ----
cat > framework/domains/etl/validators.py <<'EOF'
import numpy as np

def null_rate_ok(arr, tolerance):
    arr = np.array(arr, dtype=object)
    rate = np.mean(arr == None)
    return rate <= tolerance, rate
EOF

cat > tests/etl/test_tolerances.py <<'EOF'
import pytest, numpy as np
from framework.core.config_loader import load_config
from framework.domains.etl.validators import null_rate_ok
from framework.core.allure_utils import attach_text

@pytest.mark.etl
def test_null_rate(request):
    cfg = load_config(request.config.getoption("--env"))
    ok, rate = null_rate_ok([1,2,3,None], cfg["tolerances"]["null_rate"])
    attach_text("null_rate", str(rate))
    assert ok

@pytest.mark.parametrize("values", [
    [1,2,3,4],
    [10,11,10,11]
])
@pytest.mark.etl
def test_null_rate_param(values, request):
    cfg = load_config(request.config.getoption("--env"))
    ok,_ = null_rate_ok(values + [None], cfg["tolerances"]["null_rate"])
    assert ok

def test_etl_diff_artifact():
    before = [1,2,3]
    after = [1,2,4]
    attach_text("etl_diff", f"Before={before}\\nAfter={after}")
EOF

# ---- ML ----
cat > framework/domains/ml/model.py <<'EOF'
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_register(X, y):
    m = LinearRegression().fit(X, y)
    mse = mean_squared_error(y, m.predict(X))
    return {"model": m, "mse": mse}
EOF

cat > tests/ml/test_model_registry.py <<'EOF'
import numpy as np, pytest
from framework.domains.ml.model import train_and_register
from framework.core.allure_utils import attach_plot

@pytest.mark.ml
def test_model():
    X = np.array([[1],[2],[3]])
    y = np.array([2,4,6])
    out = train_and_register(X,y)
    attach_plot("training_curve", [1,2,3], [2,4,6])
    assert out["mse"] < 1e-6
EOF

# ========================================================
# WORKFLOWS
# ========================================================
cat > tests/workflows/test_end_to_end.py <<'EOF'
import asyncio, numpy as np
from framework.core.config_loader import load_config
from framework.domains.api.client import get_ip
from framework.domains.etl.validators import null_rate_ok
from framework.domains.ml.model import train_and_register

def test_chained(request):
    cfg = load_config(request.config.getoption("--env"))

    data = asyncio.run(get_ip(cfg["base_url"]))
    assert "origin" in data

    ok,_ = null_rate_ok([1,2,3,None], cfg["tolerances"]["null_rate"])
    assert ok

    out = train_and_register(np.array([[1],[2],[3]]), np.array([2,4,6]))
    assert out["mse"] < 1e-4
EOF

# ========================================================
# SMOKE
# ========================================================
cat > tests/test_smoke.py <<'EOF'
import pytest

@pytest.mark.smoke
def test_boot():
    assert True
EOF

# ========================================================
# DOCKER + CI
# ========================================================
cat > docker/Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["pytest","-vv"]
EOF

cat > ci/pipeline.yaml <<'EOF'
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -vv
  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t framework:latest -f docker/Dockerfile .
EOF

# ========================================================
# REQUIREMENTS
# ========================================================
cat > requirements.txt <<'EOF'
pytest
allure-pytest
httpx
jsonschema
pydantic
scikit-learn
joblib
numpy
matplotlib
EOF

# ========================================================
# PACKAGING
# ========================================================
cat > pyproject.toml <<'EOF'
[project]
name = "unified-automation-framework"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["framework"]
EOF

echo "All set — ready to run"
echo "pytest -m ui"
echo "pytest -m api"
echo "pytest -m etl"
echo "pytest -m ai_ml"

