NVCC ?= nvcc
ARCH ?= sm_89
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --use_fast_math -Isrc

sgemm: src/main.cu src/kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ src/main.cu -lcublas

run: sgemm
	./sgemm

profile: sgemm
	./scripts/profile_ncu.sh

clean:
	rm -f sgemm results.csv profiles/*.ncu-rep

.PHONY: run profile clean
