# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the pyproject.toml file into the container at /code
COPY ./pyproject.toml /code/pyproject.toml

# Install the project and its dependencies using pyproject.toml
# --no-cache-dir: Disables the cache to keep image size down
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the application code (the 'app' directory contents)
# into the container at /code/app
# Adjust if your Python code is not in an 'app' subfolder
COPY ./app /code/app
# If main.py is at the root with static/ and templates/
# COPY ./main.py /code/main.py
# COPY ./static /code/static
# COPY ./templates /code/templates
# Ensure your paths match your project structure. Assuming main.py is in 'app/'

# Make port 8000 available to the world outside this container
# Hugging Face Spaces expects the app to listen on port 7860 by default,
# but Docker apps often use 8000. We can map this in Space config if needed,
# or change the port here and in the CMD. Let's use 8000 for now.
EXPOSE 8000

# Define environment variable (optional, can be set in HF Secrets)
# ENV NAME World

# Run main.py when the container launches
# Use uvicorn to run the FastAPI application
# --host 0.0.0.0: Makes the server accessible externally
# --port 8000: The port the server will listen on
# app.main:app: Tells uvicorn where to find the FastAPI app instance
# (in main.py inside the 'app' directory, the instance named 'app')
# Adjust 'app.main:app' if your file/instance names are different.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
