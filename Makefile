NVCC=nvcc
BIN_NAME=bfs
LDLIBS=
TARGET=main
CCFLAGS=-Wextra -Wall
CUDA_COMPILER_OPTIONS=$(addprefix --compiler-options ,${CCFLAGS}) --std=c++11
ALL_CCFLAGS=
ALL_LDFLAGS=
GENCODE_FLAGS=--gpu-architecture=compute_35
INCLUDES=-I/usr/local/cuda/samples/common/inc

.PHONY: all clean

all: main

DEBUG=
debug: DEBUG+=-g -G
debug: main

OBJECTS = bfs_cpu.o bfs_kernels.o bfs.o main.o csr_matrix.o

bfs_cpu.o: bfs_cpu.cpp csr_matrix.h common.h bfs_cpu.h 
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $< -o $@

bfs_kernels.o: bfs_kernels.cu bfs_kernels.cuh
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $< -o $@

bfs.o: bfs.cu bfs.cuh common.h
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $< -o $@

main.o: main.cpp csr_matrix.h bfs.cuh bfs_cpu.h common.h
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $< -o $@

csr_matrix.o: csr_matrix.cpp csr_matrix.h
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $< -o $@

main: ${OBJECTS}
	${NVCC} $^ ${GENCODE_FLAGS} -o $@

clean:
	rm ${TARGET} *.o
