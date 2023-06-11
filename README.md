# adverse-event-classification

This repository includes the source code used for my Master’s thesis project in Computer Science: Detecting Venous Catheters and Infections through Classification of Norwegian Adverse Event Notes: A Feasibility Study.

This project used a conda environment for handling all the packages.

The project is based on version 3.8 of Python. The latest version of Python is not compatible with all the modules used in this project (specifically the OBT).

## Required Packages

### Pandas (and Numpy, automatically installed by Pandas)

```bash
conda install -c conda-forge pandas
```

### Matplotlib

```bash
conda install -c conda-forge matplotlib
```

### seaborn

```bash
conda install -c conda-forge seaborn
```

### mendelai-brat-parser

No conda install available. Use pip.

```bash
pip install mendelai-brat-parser
```

### smart_open

Needed for mendelai-brat-parser to work.

```bash
conda install -c conda-forge smart_open
```

### scikit-learn

```bash
conda install -c conda-forge scikit-learn
```

### nltk

```bash
conda install -c conda-forge nltk
```

### iterative-stratification

For stratification of multi-label data. Repo: https://github.com/trent-b/iterative-stratification

```bash
pip install iterative-stratification
```

## Oslo-Bergen Tagger (OBT)

The installation of the OBT is a more tedious process than installing the other modules. It includes three main modules. The OBT must be correctly installed to use the accompanying python module.

### Original OBT

Git repo to OBT must be cloned into the directory, followed by setting the environment variable OBT_PATH.

```bash
git clone https://github.com/noklesta/The-Oslo-Bergen-Tagger.git
```

To see conda env variables:

```bash
conda env config vars list
```

To add a conda env variable:

```bash
conda env config vars set OBT_PATH=/path/to/directory/The-Oslo-Bergen-Tagger
```

Can see that variable has been set by `echo $OBT_PATH` or `conda env config vars list`.

### Install Python library

No conda install available. Use pip.

```bash
pip install obt
```

### Install multitagger

Need to clone into the mtag repo *inside the The-Oslo-Bergen-Tagger repo* to access the multitagger. This repository is needed to run the script `[tag-bm.sh](http://tag-bm.sh)` which is used in the python library.

```bash
git clone https://github.com/textlab/mtag.git
```

### Install OBT stat

Need to clone into OBT repo *inside the The-Oslo-Bergen-Tagger* repo.

```bash
git clone https://github.com/textlab/OBT-Stat.git
```

### Install ruby (needed for obt script to work)

```bash
conda install -c conda-forge ruby
```

### VISL CG3

VISL CG3 is needed for the OBT script to work. It has to be installed using specific instructions.

GitHub repo: [https://github.com/GrammarSoft/cg3](https://github.com/GrammarSoft/cg3) 

Instructions for install: [https://visl.sdu.dk/cg3/chunked/installation.html](https://visl.sdu.dk/cg3/chunked/installation.html).
