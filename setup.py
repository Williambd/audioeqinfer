from setuptools import setup, find_packages

setup(
    name='flikinger_eq',
    version='0.1.0',
    description='Probabilistic Equalization Discovery inspired by Flikinger',
    author='[Your Name]',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "torch",           # (if you are using normalizing flows in f_X)
        "nflows",          # (if using nflows for f_X model)
        "patsy",           # (for basis spline construction)
        "pedalboard"       # (if you're using AudioFile)
    ],
    python_requires=">=3.8",
)
