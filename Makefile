.PHONY: help install test bench format lint typecheck ci clean

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
	@echo "make clean      - Remove arquivos de build, cache e pycache"
	@echo "--------------------------------------------------------"

install:
	uv pip install -e ".[accel,dev]"
	uv run pre-commit install

test:
	TMPDIR=/models/desenvolvimento/tmp uv run pytest tests -m "not slow" -q --basetemp=/models/desenvolvimento/tmp

test-all:
	TMPDIR=/models/desenvolvimento/tmp uv run pytest tests -q --basetemp=/models/desenvolvimento/tmp

bench:
	TMPDIR=/models/desenvolvimento/tmp FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
	uv run python benchmarks/bench_pp_tg.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
		--tokens 16384 --decode 128 --repeats 3 --label guard \
		--serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive"

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

typecheck:
	uv run mypy python/freetoken

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
