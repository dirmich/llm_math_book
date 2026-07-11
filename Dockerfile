FROM python:3.10-slim

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Python packages
RUN pip install --no-cache-dir -e ".[notebook]"

# Add src to Python path
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}

# Jupyter port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=127.0.0.1", "--port=8888", "--no-browser", "--allow-root"]
