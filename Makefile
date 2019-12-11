NVCC=nvcc
LDLIBS=
TARGET=gpu-bfs
DEBUG=
CCFLAGS=-Wextra -Wall -Wunused-parameter
CUDA_COMPILER_OPTIONS=$(addprefix --compiler-options ,${CCFLAGS}) --std=c++11 -dc
ALL_CCFLAGS=
ALL_LDFLAGS=
GENCODE_FLAGS=--gpu-architecture=compute_35
INCLUDES=-I/opt/cuda/samples/common/inc -I/usr/local/cuda/samples/common/inc

.PHONY: all clean debug

all: ${TARGET}

debug: DEBUG+=-G
debug: ${TARGET}

OBJECTS = main.o csr_matrix.o bfs.o bfs_cpu.o bfs_kernels.o

bfs_cpu.o: bfs_cpu.cpp csr_matrix.h common.h bfs_cpu.h 
	${NVCC} ${INCLUDES} ${DEBUG} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

bfs_kernels.o: bfs_kernels.cu bfs_kernels.cuh
	${NVCC} ${INCLUDES} ${DEBUG} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

bfs.o: bfs.cu bfs.cuh common.h bfs_kernels.cuh
	${NVCC} ${INCLUDES} ${DEBUG} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

main.o: main.cpp csr_matrix.h bfs.cuh bfs_cpu.h common.h
	${NVCC} ${INCLUDES} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

csr_matrix.o: csr_matrix.cpp csr_matrix.h
	${NVCC} ${INCLUDES} ${DEBUG} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

${TARGET}: ${OBJECTS}
	${NVCC} ${GENCODE_FLAGS} $^ -o $@

clean:
	rm -Iv ${TARGET} *.o
