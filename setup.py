"""Installation script for the RAPID IsaacLab package."""
from pathlib import Path

from setuptools import setup, find_packages

HERE = Path(__file__).resolve().parent

setup(
    name="rapid_isaaclab",
    version="1.0.0",
    description="RAPID: Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    package_data={
        "rapid_isaaclab": ["**/*.usd", "**/*.yaml", "**/*.toml"],
        "rapid": ["**/*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
)
