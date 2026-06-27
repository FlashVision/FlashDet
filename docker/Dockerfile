FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY flashdet/ flashdet/

RUN pip install --no-cache-dir -e ".[all]"

COPY train.py test.py ./
COPY scripts/ scripts/
COPY examples/ examples/

ENTRYPOINT ["flashdet"]
CMD ["--help"]
