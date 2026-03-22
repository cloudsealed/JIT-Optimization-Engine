from setuptools import setup, find_packages

setup(
    name="hpmce-core",
    version="1.0.0",
    author="Lead Engineer",
    description="High-Performance Engine for Mission-Critical Fiscal Telemetry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "numba",
        "requests"
    ],
    python_requires='>=3.8',
)