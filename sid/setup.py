from setuptools import find_packages, setup


setup(
    name="sid-assistant",
    version="2.0.0",
    description="Sid - Fully offline autonomous desktop assistant",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "pydantic==2.10.3",
        "httpx==0.27.2",
        "psutil==6.1.1",
        "apscheduler==3.10.4",
    ],
    entry_points={"console_scripts": ["sid=main:main"]},
)
