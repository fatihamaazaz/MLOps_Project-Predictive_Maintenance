import setuptools


__version__ = "0.0.0"

REPO_NAME = "MLOps_Project"
AUTHOR_USER_NAME = "fatihamaazaz"
SRC_REPO = "Classifiers"
AUTHOR_EMAIL = "fatihamaazaz@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for MLOps project",
    long_description="Setup of MLOps project named predictive maintenance",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)