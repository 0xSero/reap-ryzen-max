#!/usr/bin/env bash
# Bootstrap CerebrasResearch/reap on AMD Ryzen AI MAX+ 395 (Strix Halo).
#
# Installs PyTorch from TheRock's gfx1151 nightly index, clones the REAP repo
# from my fork's PR branch (CerebrasResearch/reap#22) until upstream merges,
# installs it editable, and snapshots Nemotron-3 with the modeling patches
# applied.
#
# Re-running is safe; everything checks for prior state before doing work.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REAP_DIR="${ROOT}/reap"
REAP_REPO="${REAP_REPO:-https://github.com/0xSero/reap.git}"
REAP_BRANCH="${REAP_BRANCH:-nemotron3-and-strix-halo}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-${PYTHON} -m pip}"

# 1. PyTorch nightly for gfx1151. Skip if torch already imports CUDA OK.
if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[bootstrap] installing torch from rocm.nightlies.amd.com/v2/gfx1151/"
    ${PIP} install --pre --upgrade torch \
        --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
else
    echo "[bootstrap] torch already importable with cuda available, skipping reinstall"
fi

"${PYTHON}" -c "import torch; print(f'[bootstrap] torch={torch.__version__} hip={torch.version.hip} cuda_available={torch.cuda.is_available()}')"

# 2. Clone REAP (my PR branch until upstream merges).
if [ ! -d "${REAP_DIR}/.git" ]; then
    echo "[bootstrap] cloning ${REAP_REPO} (${REAP_BRANCH}) into ${REAP_DIR}"
    git clone --branch "${REAP_BRANCH}" "${REAP_REPO}" "${REAP_DIR}"
else
    echo "[bootstrap] ${REAP_DIR} exists, fetching latest"
    git -C "${REAP_DIR}" fetch origin "${REAP_BRANCH}"
    git -C "${REAP_DIR}" checkout "${REAP_BRANCH}"
    git -C "${REAP_DIR}" pull --ff-only origin "${REAP_BRANCH}"
fi

# 3. Editable install + base deps.
echo "[bootstrap] pip installing reap (editable)"
${PIP} install -e "${REAP_DIR}"

# 4. Snapshot Nemotron-3 and apply the REAP modeling patches.
ARTIFACT="${REAP_DIR}/artifacts/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
if [ ! -f "${ARTIFACT}/config.json" ]; then
    echo "[bootstrap] running patch_nemotron_h.py (this downloads ~60 GiB)"
    (cd "${REAP_DIR}" && "${PYTHON}" scripts/patch_nemotron_h.py)
else
    echo "[bootstrap] ${ARTIFACT} already populated, skipping snapshot"
    # Re-copy the patched modeling file in case the upstream commit was updated.
    cp -v "${REAP_DIR}/src/reap/models/modeling_nemotron_h.py" \
          "${ARTIFACT}/modeling_nemotron_h.py"
fi

echo
echo "[bootstrap] done. next:"
echo "    bash run.sh"
