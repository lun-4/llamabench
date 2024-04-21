# llamabench
end-to-end benchmarking script for llama.cpp

## how

first, get llama.cpp setup

```sh
git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
# using this commit for my llama3 benchmarks
git checkout 5cf5e7d490dfdd2e70bface2d35dfd14aa44b4fb

# verify it builds
make -j8

# verify it builds with openblas
make LLAMA_OPENBLAS=1 -j8

# verify it builds with Vulkan (optional when benching a system)
# requires vulkan headers, vulkan icd(??), vulkan drivers(??)
make LLAMA_VULKAN=1 -j8
```

then, get this repo setup. all done in pure python3, no pip required

```sh
git clone https://github.com/lun-4/llamabench
cd llamabench

# using this model for benches
wget 'https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true'
# not required but prettier names = more aesthetic
mv 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true' 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'


# actually run the bench
env MAXTHREADS=12 MAKEFLAGS=-j8 LLAMACPP=/path/to/directory/llama.cpp MODEL=/path/to/model/file/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf python3 ./bench.py

# a lot of data is spit out to stdout/stderr, capture everything to a file
```

(TODO: process data for pretty visualizations)
