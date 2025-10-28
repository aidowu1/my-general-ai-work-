#!/bin/bash
set -e

# Write helper python script that prints xgboost include and lib paths
cat > /workspace/get_xgboost_paths.py <<'PY'
import xgboost, os, sys
import importlib.util
import site
sp = site.getsitepackages() if hasattr(site, "getsitepackages") else [site.getusersitepackages()]
# Module package dir
pkg_dir = os.path.dirname(xgboost.__file__)
# attempt to find libxgboost.* and include/xgboost/c_api.h
lib_paths = []
for d in sp + [pkg_dir]:
    for cand in [d, os.path.join(d, "xgboost"), os.path.join(d, "lib")]:
        if os.path.isdir(cand):
            for fname in ["libxgboost.so", "libxgboost.dylib", "xgboost.dll"]:
                p = os.path.join(cand, fname)
                if os.path.exists(p):
                    lib_paths.append(os.path.abspath(p))
# Find include (headers bundled in package)
include_path = os.path.join(pkg_dir, "include") if os.path.isdir(os.path.join(pkg_dir, "include")) else pkg_dir
print(include_path)
if lib_paths:
    print(lib_paths[0])
else:
    # fallback: try to search package dir for libxgboost
    for root, dirs, files in os.walk(pkg_dir):
        for f in files:
            if f.startswith("libxgboost"):
                print(os.path.join(root, f)); sys.exit(0)
    # if not found, print empty second line
    print("")
PY

chmod +x /workspace/get_xgboost_paths.py

cat > /workspace/README.devcontainer.md <<'TXT'
Devcontainer created.

Important commands to run in the container:

1) Show detected xgboost include and lib:
   python3 get_xgboost_paths.py

2) Train model (produces model JSON and test CSV):
   python3 train_xgb_diabetes.py

3) Configure CMake and build C++ inference:
   ./build_and_run.sh

TXT

echo "postCreateCommand completed - helper scripts created."
