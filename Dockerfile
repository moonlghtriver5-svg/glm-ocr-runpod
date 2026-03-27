FROM vllm/vllm-openai:nightly

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y transformers || true \
 && pip install -U git+https://github.com/huggingface/transformers.git \
 && pip install -U git+https://github.com/zai-org/glm-ocr.git \
 && pip install runpod requests pillow

# Model downloads at runtime on first boot, cached on network volume
ENV HF_HOME=/runpod-volume/huggingface
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm-cache
ENV MAX_MODEL_LEN=16384
ENV GPU_MEMORY_UTILIZATION=0.95
ENV SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
ENV ENFORCE_EAGER=0
ENV MAX_IMAGE_SIDE=2000
ENV USE_GLMOCR_SDK=1

RUN mkdir -p /root/.config/glm-ocr \
 && printf "pipeline:\n  maas:\n    enabled: false\n  ocr_api:\n    api_host: localhost\n    api_port: 8080\n" > /root/.config/glm-ocr/config.yaml

COPY handler.py /handler.py

EXPOSE 8080

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
