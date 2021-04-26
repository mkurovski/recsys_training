# recsys_training

Recommender System Training Package

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mkurovski/recsys_training/master)

## Description

Hands-on Training for Recommender Systems developed for Machine Learning Essentials 2020.

## Installation

In order to set up the necessary environment:

1. create an environment `recsys_training` with the help of [conda],
   
   ```
   conda env create -f environment.yaml
   ```
   
2. activate the new environment with
   
   ```
   conda activate recsys_training
   ```
   
3. install `recsys_training` with:
   
   ```
   python setup.py install # or develop
   ```

### Docker

Make sure you have `docker` and `docker-compose` installed.

1. Build the image with using the `Dockerfile` in `docker`
    ```
    docker build -t recsys-training:mle -f Dockerfile .
    ```

2. Start the container with `docker-compose` pointing to the yaml-file
    ```
    docker-compose up -f docker/docker-compose.yaml
    ```
    
The jupyter lab port `8888` will be mapped to the same port on your host machine, simply got to your preferred browser and enter via
    ```
    http://localhost:8888/
    ```

## Usage

There are 9 notebooks within `notebooks/` each starting with a number followed by `_e_` for exercise. Within `notebooks/solutions/`you will find all notebooks with a solution proposal implemented. It is strongly advised to go through the notebooks in numerically ascending order.

We use MovieLens 100k as example dataset for the lessons.

* Go to [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/) and download the `.zip` file.
* Move it into `data/raw/` and unzip it there.
* Now you are all set (assuming you have a working browser) - lean back and enjoy the ride ;)

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
