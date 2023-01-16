import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def parse_requirements_file(fname):
    requirements = list()
    with open(fname, 'r') as fid:
        for line in fid:
            req = line.strip()
            if req.startswith('#'):
                continue
            # strip end-of-line comments
            req = req.split('#', maxsplit=1)[0].strip()
            requirements.append(req)
    return requirements

#install_requires = parse_requirements_file("requirements.txt")

install_requires=[
    'numpy',
    'sklearn',
    'blockmatrix'
]

setuptools.setup(
    name="pyclf-bci",
    version="0.0.2",
    author="Simon Kojima",
    author_email="simon.kojima@outlook.com",
    description="An implementation of various classifier for bci application.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonkojima/pyclf",
    project_urls={
        "Bug Tracker": "https://github.com/simonkojima/pyclf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = install_requires,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)