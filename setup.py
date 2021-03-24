import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timesias",
    version="0.0.1",
    author="Hanrui Zhang, Yuanfang Guan",
    author_email="rayezh@umich.edu, gyuanfan@umich.edu",
    description="A machine-learning framework for predicting outcomes from time-series history.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuanLab/timesias",
    project_urls={
        "Bug Tracker": "https://github.com/GuanLab/timesias/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where = 'src'),
    entry_points = {
        'console_scripts': [
            'timesias = src.__main__:main'
            ]
        },
    python_requires=">=3.6",
)
