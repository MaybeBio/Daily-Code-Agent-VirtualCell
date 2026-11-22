FROM nvcr.io/nvidia/pytorch:22.12-py3
WORKDIR /workspace/immuneworld
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-deps .
ENTRYPOINT ["immuneworld-train"]
