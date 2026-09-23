.PHONY: help install test test-safe bench bench-flash format lint typecheck ci clean preflight profile crossref

export CUDA_HOME ?= /models/outros/cuda-13.3
export CUDA_PATH ?= $(CUDA_HOME)
export PATH := /usr/lib/ccache:$(CUDA_HOME)/bin:/home/natal/.local/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)
export CUDACXX ?= $(CUDA_HOME)/bin/nvcc
export CMAKE_CUDA_COMPILER ?= $(CUDA_HOME)/bin/nvcc
export CMAKE_CUDA_ARCHITECTURES ?= 120;120a
export TORCH_CUDA_ARCH_LIST ?= 12.0;12.0a
export TVM_FFI_CUDA_ARCH_LIST ?= 12.0 12.0a
export MAX_JOBS ?= 24
export CMAKE_BUILD_PARALLEL_LEVEL ?= 24
export MAKEFLAGS ?= -j24
export OMP_NUM_THREADS ?= 24
export RAYON_NUM_THREADS ?= 24
export FREETOKEN_KERNEL_CACHE_JOBS ?= 24
export CFLAGS ?= -O3 -march=native
export CXXFLAGS ?= -O3 -march=native
export CUDAFLAGS ?= -O3 --use_fast_math -Xfatbin=-compress-all
export TMPDIR ?= /models/desenvolvimento/tmp
TIMEOUT ?= 60

# Default target
help:
	@echo "FreeToken Next - Comandos Automáticos"
	@echo "--------------------------------------------------------"
	@echo "make install    - Instala dependências e hooks locais"
	@echo "make test       - Roda a suite de testes unitários principal"
	@echo "make test-all   - Roda todos os testes (incluindo os lentos)"
	@echo "make bench      - Roda o benchmark PP/TG padrão com 35B-A3B"
	@echo "make format     - Formata o código com ruff"
	@echo "make lint       - Checa problemas de estilo com ruff"
	@echo "make typecheck  - Faz a análise estática de tipos com MyPy"
	@echo "make ci         - Emula uma esteira de CI local (Lint + Typecheck + Tests)"
	@echo "make preflight  - Limpa zumbis, /tmp e checa VRAM/RAM antes de testes pesados"
	@echo "make profile    - Benchmark com validacao automatica vs anchors PERFORMANCE.md"
	@echo "make crossref   - Valida referencias cruzadas entre docs/dev/*.md"
	@echo "make rebuild    - Limpa caches e recompila as extensoes C++ do zero"
	@echo "make clean      - Remove arquivos de build, cache e pycache"
	@echo "--------------------------------------------------------"

install:
	uv pip install -e ".[accel,dev]"
	uv run --no-sync pre-commit install

preflight:
	@./scripts/preflight.sh

rebuild:
	@echo "Limpando artefatos antigos de C++..."
	rm -rf build/ python/freetoken.egg-info/
	@echo "Recompilando extensões..."
	uv run --no-sync python setup.py build_ext --inplace

test:
	TMPDIR=/models/desenvolvimento/tmp uv run --no-sync pytest tests -m "not slow" -q --basetemp=/models/desenvolvimento/tmp

test-safe:
	TMPDIR=/models/desenvolvimento/tmp uv run --no-sync python scripts/test-runner.py --timeout $(TIMEOUT) -- uv run --no-sync pytest tests -m "not slow" -q --basetemp=/models/desenvolvimento/tmp

test-all:
	TMPDIR=/models/desenvolvimento/tmp uv run --no-sync pytest tests -q --basetemp=/models/desenvolvimento/tmp

bench:
	TMPDIR=/models/desenvolvimento/tmp FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
	uv run --no-sync python benchmarks/bench_pp_tg.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
		--tokens 16384 --decode 128 --repeats 3 --label guard \
		--serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive"

MTP ?= 0
MTP_ARG := $(if $(filter-out 0,$(MTP)),--serve-arg="--spec-mtp $(MTP)")
bench-turbo4:
	TMPDIR=/models/desenvolvimento/tmp FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
	uv run --no-sync python benchmarks/bench_pp_tg.py --model /models/Qwen3.8-Flash-Next-NVFP4-Radix \
		--tokens 16384 --decode 16 --repeats 3 --label turbo4-mtp$(MTP) --mem-ratio 0.9 \
		--serve-arg="--num-tokens 16576" --serve-arg="--kv-format turbo4" \
		--serve-arg="--cache-type naive" $(MTP_ARG)

# Flash-Next GGUF (IQ4_XS + MTP): 4K prompt, 256 tokens, 16K capacity, turbo3. Repeats hit the
# radix prefix cache, so its PP is cached-prefix PP; cold PP needs a unique prompt per request.
bench-flash:
	TMPDIR=/models/desenvolvimento/tmp FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
	uv run --no-sync python benchmarks/bench_pp_tg.py \
		--model /models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS \
		--tokens 4096 --decode 256 --repeats 3 --label flash-gguf-mtp$(MTP) \
		--serve-arg="--max-seq-len-override 16384" --serve-arg="--kv-format turbo3" \
		--serve-arg="--cuda-graph-max-bs 1" --serve-arg="--max-running-requests 1" $(MTP_ARG)

format:
	uv run --no-sync ruff format .
	uv run --no-sync ruff check --fix .

lint:
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy python/freetoken

ci:
	@echo "--- Rodando CI Local ---"
	make format
	make lint
	make typecheck
	make test

clean:
	rm -rf build/ dist/ .pytest_cache/ freetoken-kernel-cache/build/ .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

profile:
	@./scripts/bench-profile.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT --tokens 16384 --decode 128 --repeats 3 --label profile --serve-args "--num-tokens 16576 --cache-type naive"

mtp-equiv:
	@./scripts/bench-mtp-equiv.sh --model $(MODEL) --mtp-k $(K) $(MTP_ARGS)

crossref:
	@./scripts/validate-crossref.py

doc-append:
	@./scripts/doc-append.py $(ARGS)

doc-test:
	@./scripts/doc-append.py --lessons "Teste auto-doc -> script funciona -> doc atualizada"
	@./scripts/doc-append.py --errors "TEST-001 | Teste doc-append | script testado | validado | make doc-test | scripts/doc-append.py"
