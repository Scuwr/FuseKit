set -Eeuo pipefail

ENV_NAME="pypi-env"
PROJ_ROOT="/storage/cmarnold/projects"
FUSEKIT="${PROJ_ROOT}/FuseKit"

# Make sure user site-packages doesn't interfere if activation ever fails
export PYTHONNOUSERSITE=1

# 1) Ensure conda is on PATH & enable 'conda activate' for this shell
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH"; exit 1
fi
eval "$("$(conda info --base)"/bin/conda shell.bash hook)"

# 2) Clean slate
conda deactivate || true
conda env remove -n "${ENV_NAME}" -y || true

# 3) Create & activate env
conda create -n "${ENV_NAME}" python=3.11 pip -y
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install fusekit

cd "${FUSEKIT}/tests"
./python_tests.sh