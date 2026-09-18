# ==================================================================
# Dockerfile para FreeToken-Next
# ==================================================================
# Congela o ambiente exato de execução do motor de inferência:
# Ubuntu 22.04, Python 3.12, CUDA 12.x/13.x compatível.
# ==================================================================

# Base Oficial Nvidia CUDA (Desenvolvimento para ter nvcc)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Define variáveis de ambiente essenciais
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/root/.uvcache
ENV TMPDIR=/freetoken/tmp

# Instala ferramentas base do sistema e Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configura o Python 3.12 como padrão
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Instala uv (Gerenciador de pacotes moderno)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Cria a pasta do projeto
WORKDIR /freetoken

# Copia apenas os arquivos de configuração para aproveitar cache de pacotes
COPY pyproject.toml setup.py uv.lock Makefile ./

# Cria o ambiente virtual e instala as dependências via uv
RUN uv venv /freetoken/.venv && \
    uv pip install -e ".[accel]"

# Copia o restante do código-fonte do motor
COPY . .

# Compila as extensões C++ (pinned_tensor, cpu_moe, ple_store)
RUN /freetoken/.venv/bin/python setup.py build_ext --inplace

# Expõe as portas do servidor HTTP
EXPOSE 8081

# Garante que as pastas temporárias existam
RUN mkdir -p /freetoken/tmp

# Comando padrão
CMD ["/freetoken/.venv/bin/ft", "serve", "--help"]
