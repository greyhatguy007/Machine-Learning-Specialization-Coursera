# Machine Learning Specialization Coursera

![title](/resources/title-head.png)

## Introduction

Contains Solutions and Notes for the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction/?utm_medium=coursera&utm_source=home-page&utm_campaign=mlslaunch2022IN) by Andrew Ng on Coursera.

This repo is folked from [here](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera)
with the following modifications:

1. Rewrite README.md, adding Local Environment Setup guide
2. Add `requirements.txt` (#) and `.gitignore`.

## Why Yet Another Folk?

Because:

(#) Ironically this should have been of top priority
since people cloning this repo will want to run their
notebooks locally. Unfortunately, I have had a hard
time figuring out the exact versions of the
packages/dependencies that the notebooks require.
**If wrong versions are installed, some notebooks**
**will yield errors that should not have been there!**

And I don't want you to also suffer from this
version mess.

Just install with that `requirements.txt`
and you will be good to go (see below).

## Local Environment Setup

The following guide applies to Linux. Windows users
should be able to achieve the same goals, just in
a different way.

### Install NVIDIA CUDA libraries

See this guide: <https://twm.me/posts/how-to-use-tensorflow-with-gpu-linux/>

### Install Python

You **MUST** have Python 3.7 installed. Otherwise,
when you get to the part of installing packages,
conflicts will occur and you won't be able to resolve
them!

1. Install Python 3.7.17. See [this guide](https://tecadmin.net/how-to-install-python-3-12-on-ubuntu-debian-linuxmint/)
    (just substitute Python 3.12 with Python 3.7.17)

2. Verify your Python 3.7 installation.

    ```sh
    python3.7 -V
    pip3.7 -V
    ```

### Setup your Virtual Environment

First, `cd` into this project's directory. Then:

```sh
virtualenv venv -p $(which python3.7)
source ./venv/bin/activate
pip install -r requirements.txt
```

Then, open the notebook files with VSCode.
You may need to install relevant extensions
like Jupyter Notebook for VSCode first.

### Ready

From now on, you don't have to repeat the
previous steps anymore. Just open the notebooks
and you should be able to practice right away!

### Known Issues

Only if you encounter these issues will you need
to fix them!

- [Jupyter Kernel crashes when using Tensorflow](https://github.com/microsoft/vscode-jupyter/wiki/Kernel-crashes-when-using-tensorflow)
- [Why you MAY need to install `tensorrt`](https://stackoverflow.com/a/75745465/13680015)
