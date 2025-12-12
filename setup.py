"""Setup script for RL Soccer Arena."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rl-soccer-arena",
    version="1.0.0",
    author="RL Soccer Arena Team",
    description="Professional 3D Soccer Simulation with Reinforcement Learning and Self-Play",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl-soccer-arena",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.12.1",
            "mypy==1.8.0",
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "flake8==7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soccer-train=scripts.train:main",
            "soccer-evaluate=scripts.evaluate:main",
            "soccer-visualize=scripts.visualize:main",
        ],
    },
)
