# Use a base image with the necessary dependencies
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the PolyPhy source code to the working directory
COPY . .

# Install apt dependencies
RUN apt update 
RUN apt install git -y

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install . -U

# Expose port 8000 for the web server after tox
EXPOSE 8000

# Expose port 8888 for the jupyter notebook
EXPOSE 8888

# Set the default command to start
CMD ["bash"]
