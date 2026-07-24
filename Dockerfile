# Use the lightweight Alpine image
FROM docker.io/python:3.14.6-alpine3.23

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for Pillow and general Python C-extensions on Alpine
RUN apk update && \
    apk add --no-cache \
    gcc \
    musl-dev \
    jpeg-dev \
    zlib-dev \
    freetype-dev \
    libffi-dev

# Copy the requirements file (see step 2 below)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY app.py api_client.py config.py database.py ./

# Patch app.py to bind to 0.0.0.0 instead of 127.0.0.1 for container access
RUN sed -i 's/server_name="127.0.0.1"/server_name="0.0.0.0"/' app.py

# Create a non-root user for security best practices
RUN addgroup -S nanogroup && adduser -S nanouser -G nanogroup

# Create the outputs directory and assign ownership of the app directory to the new user
RUN mkdir -p outputs && \
    chown -R nanouser:nanogroup /app

# Switch to the non-root user
USER nanouser

# Expose the default Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
