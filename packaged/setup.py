"""Compile the deployment core into a C-extension so the source stays hidden.

    python setup.py build_ext --inplace
"""
from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

# Keep docstrings and source-line comments out of the compiled binary.
Options.docstrings = False
Options.emit_code_comments = False

setup(
    name="pi05_deploy",
    version="0.1.0",
    packages=["pi05_deploy"],
    python_requires=">=3.10",
    # Pinned to the exact LeRobot fork (v0.4.2) the package was built against.
    # The [pi] extra also pulls the patched transformers@fix/lerobot_openpi fork.
    install_requires=[
        "lerobot[pi] @ git+https://github.com/jokeru8/piper_lerobot.git"
        "@79fcd7dfb8deace928c718dda691311230f58b25",
        "numpy==1.26.4",
        "pillow>=10.0",
    ],
    ext_modules=cythonize(
        ["pi05_deploy/_core.py"],
        compiler_directives={"language_level": "3"},
        build_dir="build",
    ),
    zip_safe=False,
)
