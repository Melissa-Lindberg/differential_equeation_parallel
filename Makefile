CXX := g++
MPICXX := mpicxx

CXXFLAGS := -std=c++11 -O2 -Wall -Wextra
OMPFLAGS := -fopenmp

SEQ_TARGET := consistent
OMP_TARGET := openmp
MPI_TARGET := mpi

SEQ_SRCS := consistent.cpp log.cpp
OMP_SRCS := OpenMP.cpp log.cpp
MPI_SRCS := MPI.cpp log.cpp

OBJ_DIR := build
SEQ_OBJ_DIR := $(OBJ_DIR)/seq
OMP_OBJ_DIR := $(OBJ_DIR)/omp
MPI_OBJ_DIR := $(OBJ_DIR)/mpi

SEQ_OBJS := $(addprefix $(SEQ_OBJ_DIR)/,$(SEQ_SRCS:.cpp=.o))
OMP_OBJS := $(addprefix $(OMP_OBJ_DIR)/,$(OMP_SRCS:.cpp=.o))
MPI_OBJS := $(addprefix $(MPI_OBJ_DIR)/,$(MPI_SRCS:.cpp=.o))

ALL_TARGETS := $(SEQ_TARGET) $(OMP_TARGET) $(MPI_TARGET)

.PHONY: all clean

all: $(ALL_TARGETS)

$(SEQ_TARGET): $(SEQ_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OMP_TARGET): $(OMP_OBJS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(MPI_TARGET): $(MPI_OBJS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@ $(MPI_LDFLAGS)

$(SEQ_OBJ_DIR)/%.o: %.cpp | $(SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OMP_OBJ_DIR)/%.o: %.cpp | $(OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(MPI_OBJ_DIR)/%.o: %.cpp | $(MPI_OBJ_DIR)
	$(MPICXX) $(CXXFLAGS) $(MPI_CXXFLAGS) -c $< -o $@

$(SEQ_OBJ_DIR):
	@mkdir -p $@

$(OMP_OBJ_DIR):
	@mkdir -p $@

$(MPI_OBJ_DIR):
	@mkdir -p $@
	
clean:
	rm -rf $(OBJ_DIR) $(SEQ_TARGET) $(OMP_TARGET) $(MPI_TARGET) output
