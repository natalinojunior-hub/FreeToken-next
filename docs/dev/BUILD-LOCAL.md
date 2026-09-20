# Manual Canônico de Compilação e Otimização Local

> **Host:** AMD Ryzen 9 9900X (Zen 5, 24 threads) + NVIDIA GeForce RTX 5080 (Blackwell SM120, 16 GiB VRAM) + 96 GB DDR5 RAM  
> **Fonte Única da Verdade:** Este documento define as flags, variáveis de ambiente e regras de compilação para extrair 100% de desempenho do hardware neste nó.

---

## 1. Perfil de Hardware e Ambiente

| Componente | Especificação | Papel no Runtime / Compilação |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 9 9900X (12c/24t, Zen 5) | Compilação paralela com 24 jobs (`-j24`), kernels de CPU MoE e descompactação AVX-512. |
| **GPU** | NVIDIA GeForce RTX 5080 (84 SM, SM120) | Compute Capability 12.0 (`sm_120a`). Suporte nativo a Tensor Cores NVFP4 e FP8. |
| **RAM** | 96 GB DDR5 | Permite serial ou parallel expert-bank assembly sem risco de OOM. |
| **Toolkit CUDA** | CUDA 13.3 (`/models/outros/cuda-13.3`) | Compilador `nvcc` versão 13.3.73, headers e bibliotecas runtime `libcudart.so.13`. |
| **Compilador Host** | GCC / G++ 12.5.0 | Suporte nativo a `-march=znver5`, AVX-512 (`F`, `DQ`, `BW`, `VL`, `BF16`, `VNNI`). |

---

## 2. Invariantes de Compilação

Para garantir reprodutibilidade, paridade de benchmarks e velocidade máxima:

1. **Paralelismo Estrito: `-j24` / `MAX_JOBS=24`**  
   Todo processo de compilação (Ninja, make, cmake) deve usar rigorosamente 24 jobs para saturar os 24 threads do Ryzen 9 9900X sem gerar thrashing. Nunca use valores arbitrários como `-j2` ou `-j$(nproc)`.
2. **CUDA 13.3 Rigorosamente Obrigatório (Nenhuma Versão Inferior)**  
   O toolkit CUDA deve ser estritamente `>= 13.3` (`/models/outros/cuda-13.3`). Versões anteriores não possuem suporte nativo à arquitetura Blackwell SM120/SM120a, nem às instruções SASS para Tensor Cores NVFP4 e FP8 do SM120a.
3. **Arquitetura CUDA: `12.0;12.0a` (`sm_120;sm_120a`)**  
   Para Blackwell RTX 5080, `TORCH_CUDA_ARCH_LIST="12.0;12.0a"` e `CMAKE_CUDA_ARCHITECTURES="120;120a"`. O sufixo `a` emite instruções reais de SASS para os Tensor Cores FP4 (`compute_120a,sm_120a`), evitando overhead de JIT PTX em tempo de execução. Flags nvcc: `-O3 --use_fast_math -Xfatbin=-compress-all`.
4. **Vetorização Zen 5: `-O3 -march=native`**  
   O GCC 12.5.0 resolve `-march=native` diretamente como `-march=znver5 -mtune=znver5`, ativando o conjunto completo de instruções AVX-512 do Zen 5 (`AVX512_BF16`, `AVX512_VNNI`, `AVX512_FP16`, `VPCLMULQDQ`).

---

## 3. Variáveis de Ambiente Canônicas

Carregue essas variáveis no seu shell ou garanta que seu ambiente as possua antes de compilar ou rodar benchmarks:

```bash
export CUDA_HOME=/models/outros/cuda-13.3
export CUDA_PATH=/models/outros/cuda-13.3
export PATH="/models/outros/cuda-13.3/bin:/home/natal/.local/bin:$PATH"
export LD_LIBRARY_PATH="/models/outros/cuda-13.3/lib64:$LD_LIBRARY_PATH"
export CUDACXX=/models/outros/cuda-13.3/bin/nvcc
export CMAKE_CUDA_COMPILER=/models/outros/cuda-13.3/bin/nvcc
export CMAKE_CUDA_ARCHITECTURES="120;120a"
export TORCH_CUDA_ARCH_LIST="12.0;12.0a"
export TVM_FFI_CUDA_ARCH_LIST="12.0 12.0a"
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export MAKEFLAGS="-j24"
export OMP_NUM_THREADS=24
export RAYON_NUM_THREADS=24
export FREETOKEN_KERNEL_CACHE_JOBS=24
export CFLAGS="-O3 -march=native"
export CXXFLAGS="-O3 -march=native"
export CUDAFLAGS="-O3 --use_fast_math -Xfatbin=-compress-all"
export TMPDIR=/models/desenvolvimento/tmp
```

*(Nota: O `Makefile` do `freetoken-next` já exporta automaticamente todas essas variáveis em todos os seus alvos).*

---

## 4. Comandos de Compilação

### A. Extensões C++/CUDA do FreeToken (`setup.py`)

As extensões de alta performance incluem:
* `freetoken.kernel._pinned_tensor`: Alocador assíncrono de memória pinada com CUDA runtime.
* `freetoken.kernel._cpu_moe`: Micro-kernels de descompactação e GEMV BF16 com dispatch AVX-512.
* `freetoken.kernel._ple_store`: Camada Linux de persistência O_DIRECT / io_uring para PLE.

Para recompilar do zero com otimização total:

```bash
# Via Makefile (recomendado)
make rebuild

# Ou diretamente via uv + setup.py
uv run --no-sync python setup.py build_ext --inplace
```

### B. Módulos C/C++ Standalone / CMake (ex: `llama-turbo-optimal`)

Se compilar bibliotecas ou sidecars via CMake neste nó:

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/models/outros/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/models/outros/cuda-13.3 \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24
```

---

## 5. Verificação Pós-Build

Para assegurar que as extensões C++ foram compiladas e linkadas corretamente contra o CUDA 13.3:

```bash
# 1. Verificar símbolos e linkage com libcudart.so.13
ldd python/freetoken/kernel/_pinned_tensor.*.so | grep cuda

# 2. Rodar suite de testes de kernel
uv run --no-sync pytest tests/kernels/ -q
```
