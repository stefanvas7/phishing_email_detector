from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list[str]:
    """Reads the requirements from a given file returning it as a string."""
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="phishing_email_detector",
    version="0.0.1",
    author="Stefan Vassilev",
    author_email="stevasvas@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8"
)