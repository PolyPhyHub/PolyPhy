# Use a base image with the necessary dependencies and CUDA support
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Copy the PolyPhy source code to the working directory
COPY . .

# Install apt dependencies
RUN apt update
RUN apt install git python3-pip -y

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install . -U

# Install Vulkan
RUN apt install libvulkan1 -y

# Expose port 8000 for the web server after tox
EXPOSE 8000

# Expose port 8888 for the Jupyter notebook
EXPOSE 8888

# Set the default command to start
CMD ["bash"]
