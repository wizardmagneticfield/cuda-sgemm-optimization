# Change ARCH for your GPU: sm_89 = RTX 40xx, sm_86 = RTX 30xx, sm_75 = T4/RTX 20xx, sm_80 = A100
ARCH ?= sm_89
NVCC ?= nvcc
FLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Isrc

sgemm: src/main.cu src/kernels.cuh
	$(NVCC) $(FLAGS) -o $@ src/main.cu -lcublas

run: sgemm
	./sgemm

# Profile every kernel at N=4096 with Nsight Compute (needs ncu installed)
profile: sgemm
	./scripts/profile_ncu.sh 4096

clean:
	rm -f sgemm results.csv profiles/*.ncu-rep

.PHONY: run profile clean
