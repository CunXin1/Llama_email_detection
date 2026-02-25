FROM vllm/vllm-openai:v0.6.6.post1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 模型路径通过 docker run -v 挂载，不 COPY 进镜像
ENV MODEL_PATH=meta-llama/Llama-3.1-8B
ENV QUANTIZATION=
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TENSOR_PARALLEL_SIZE=1

EXPOSE 8000

# vllm/vllm-openai sets its own ENTRYPOINT (api_server.py); reset it so our
# uvicorn command runs as the main process instead of being passed as arguments.
ENTRYPOINT []
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
