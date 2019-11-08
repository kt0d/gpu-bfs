NVCC=nvcc
BIN_NAME=bfs
LDLIBS=
TARGET=main
DEBUG=
COMPILER_OPTIONS=-Wextra -Wall
CUDA_COMPILER_OPTIONS=$(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option)) -std=c++11
INCLUDES=/usr/local/cuda/samples/common/inc

.PHONY: all clean



all: main

debug: DEBUG+=-g 
debug: main

bfs_cpu.o: csr_matrix.h common.h bfs_cpu.h
	${NVCC} ${DEBUG} -c ${CUDA_COMPILER_OPTIONS} bfs_cpu.cpp

bfs.o: bfs.cu bfs.cuh common.h
	${NVCC} -I${INCLUDES} ${DEBUG} -c ${CUDA_COMPILER_OPTIONS} bfs.cu

main.o: main.cpp csr_matrix.h bfs.cuh bfs_cpu.h common.h
	${NVCC} ${DEBUG} -c ${CUDA_COMPILER_OPTIONS} main.cpp

csr_matrix.o: csr_matrix.cpp csr_matrix.h
	${NVCC} ${DEBUG} -c ${CUDA_COMPILER_OPTIONS} csr_matrix.cpp

main: main.o csr_matrix.o bfs.o bfs_cpu.o common.h
	${NVCC} csr_matrix.o main.o bfs.o bfs_cpu.o -o main.out

clean:
	rm ${TARGET} *.o
