FROM debian:latest

#  $ docker build . -t continuumio/miniconda:latest -t continuumio/miniconda:4.5.11 -t continuumio/miniconda2:latest -t continuumio/miniconda2:4.5.11
#  $ docker run --rm -it continuumio/miniconda2:latest /bin/bash
#  $ docker push continuumio/miniconda:latest
#  $ docker push continuumio/miniconda:4.5.11
#  $ docker push continuumio/miniconda2:latest
#  $ docker push continuumio/miniconda2:4.5.11

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y gcc unzip git curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN cd ~ && \
    git clone https://github.com/mkurovski/recsys_training.git && \
    cd recsys_training && \
    conda env create -f environment.yaml

RUN /bin/bash -c "source activate recsys_training && \
    cd ~/recsys_training && \
    python setup.py install"

# Download an unzip Data
RUN wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O ~/recsys_training/data/raw/ml-100k.zip && \
    cd ~/recsys_training/data/raw && \
    unzip ml-100k.zip

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
