#Use the official NVIDIA CUDA 12.4.1 base image for Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
LABEL authors="vatsal"

#Set environment variables to prevent interactive prompts during installation
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

#Install system dependencies, Python, and pip
RUN apt-get update &&  \
    apt-get install -y && \
    python3.10 &&  \
    python3-pip &&  \
    git &&  \
    rm -rf /var/lib/apt/lists/*

#Set python3 as the default python command
RUN ln -s /usr/bin/python3 /usr/bin/python

#Install uv, the fast Python package installer
RUN pip install uv

#Set the working directory inside the container
WORKDIR /classifier

#Copy the uv project and lock files to leverage Docker layer caching
COPY ../pyproject.toml ./
COPY ../uv.lock ./

# Install Python packages using uv sync.
# 'uv sync' is the command to install dependencies from a uv.lock file,
#ensuring a reproducible environment. '--frozen' prevents any changes to the lock file.
RUN uv sync --frozen

#Copy the rest of the project code into the working directory.
#Note: This is for the initial build. The docker-compose file will mount
#your local directory, so changes are reflected live.
COPY ../. .

#This command keeps the container running in the background indefinitely.
#You will then use 'docker exec' to get a shell inside it.
CMD ["tail", "-f", "/dev/null"]

