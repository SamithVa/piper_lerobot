#!/usr/bin/env bash
# Compile pi05_deploy/_core.py into a C-extension and assemble a source-free
# distributable in dist/. The .py source is kept here (so you can rebuild);
# only the compiled .so is copied into the shipped package.
set -e
cd "$(dirname "$0")"

# 1. Compile _core.py -> pi05_deploy/_core*.so (in place)
python setup.py build_ext --inplace

# 2. Strip ELF symbols (removes leftover symbol names from the binary)
strip --strip-all pi05_deploy/_core*.so

# 3. Assemble a source-hidden, git-URL-free package under dist/
rm -rf dist build pi05_deploy/_core.c pi05_deploy/__pycache__
mkdir -p dist/pi05_deploy
cp pi05_deploy/__init__.py dist/pi05_deploy/
cp pi05_deploy/_core*.so   dist/pi05_deploy/
cp deploy.py README.md requirements.txt dist/
# Pre-built binary wheels (lerobot fork + patched transformers) so the client
# installs with no git URL / source repo. Skipped if not yet built.
if compgen -G "wheels/*.whl" > /dev/null; then
    cp -r wheels dist/wheels
fi

echo
echo "Deliverable -> dist/"
find dist -type f | sort
