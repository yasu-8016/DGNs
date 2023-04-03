FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

RUN apt update && apt -y upgrade && apt install -y sudo git
# Install base utilities
RUN apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=anon
ARG GROUPNAME=anon
ARG UID=1000
ARG GID=1000
ARG PASSWORD=bgt54eszaq!!
ARG ENVNAME=rdk_env38

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/

# Install miniconda
ENV CONDA_DIR /home/$USERNAME/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /home/$USERNAME/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo $PASSWORD | conda init bash
RUN ( \
     echo '. /home/'$USERNAME'/conda/etc/profile.d/conda.sh'; \
     echo 'conda activate base'; \
    ) > /home/${USERNAME}/.bashrc
RUN source /home/${USERNAME}/.bashrc

# conda create
RUN conda create -n $ENVNAME python=3.8

# install conda package
SHELL ["conda", "run", "-n", "rdk_env38", "/bin/bash", "-c"]
# RUN conda install psi4 -c psi4
RUN conda install rdkit -c conda-forge
RUN conda install pandas -c conda-forge
RUN conda install pubchempy -c conda-forge
RUN conda install -c conda-forge gxx_linux-64==11.1.0

# install pip package
RUN pip3 install --upgrade pip
RUN git clone https://github.com/danielegrattarola/spektral.git
RUN cd ./spektral && pip install spektral
RUN pip install ase Ipython

RUN echo $LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USERNAME/conda/envs/$ENVNAME/lib"
RUN echo $LD_LIBRARY_PATH

