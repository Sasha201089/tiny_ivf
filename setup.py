from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="tinyivf",
    version="0.1.0",
    author_email="sashatron2010@gmail.com",
    description="Алгоритм адаптивного векторного поиска на основе SVD сжатия",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/tinyivf",
    packages=["tinyivf"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "faiss-cpu",
        "scipy"
    ],
    include_package_data=True,
    zip_safe=False,
)
