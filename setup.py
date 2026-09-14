from setuptools import (  # pyright: ignore[reportMissingModuleSource]
    find_packages,
    setup,
)

setup(
    name="hermes-agent-smart-inference",
    version="0.1.0",
    description="Small provider-neutral Smart Inference router for Hermes Agent",
    packages=find_packages(),
    python_requires=">=3.10",
)
