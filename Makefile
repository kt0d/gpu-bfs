NVCC=nvcc
BIN_NAME=bfs
LDLIBS=
TARGET=main
DEBUG=
CCFLAGS=-Wextra -Wall
CUDA_COMPILER_OPTIONS=$(addprefix --compiler-options ,${CCFLAGS}) --std=c++11
ALL_CCFLAGS=
ALL_LDFLAGS=
GENCODE_FLAGS=--gpu-architecture=compute_35
INCLUDES=-I/usr/local/cuda/samples/common/inc

.PHONY: all clean

all: main

debug: DEBUG+=-g -G
debug: main

bfs_cpu.o: bfs_cpu.cpp csr_matrix.h common.h bfs_cpu.h 
	${NVCC} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $<

bfs.o: bfs.cu bfs.cuh common.h
	${NVCC} ${INCLUDES} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $<

main.o: main.cpp csr_matrix.h bfs.cuh bfs_cpu.h common.h
	${NVCC} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $<

csr_matrix.o: csr_matrix.cpp csr_matrix.h
	${NVCC} ${DEBUG} ${CUDA_COMPILER_OPTIONS} ${GENCODE_FLAGS} -c $<

main: main.o csr_matrix.o bfs.o bfs_cpu.o
	${NVCC} $^ ${GENCODE_FLAGS} -o $@

clean:
	rm ${TARGET} *.o
