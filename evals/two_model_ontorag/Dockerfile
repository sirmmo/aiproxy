FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
WORKDIR /work
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "ragas>=0.4,<0.5" "langchain-core>=0.3,<0.4" "langchain-community>=0.3,<0.4" "langchain-openai>=0.3,<0.4" \
        "trulens-core>=2.14" "trulens-providers-openai>=2.14" "sentence-transformers>=3" "openai>=1.60,<2" pandas numpy httpx pyyaml tqdm
CMD ["python", "-c", "import ragas, trulens.core; print('ragas', ragas.__version__, 'ok')"]
