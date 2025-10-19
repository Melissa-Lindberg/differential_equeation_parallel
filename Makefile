CXX := g++
CXXFLAGS := -std=c++11 -O2 -Wall -Wextra
OMPFLAGS := -fopenmp

SEQ_TARGET := consistent
OMP_TARGET := openmp

SEQ_SRCS := consistent.cpp log.cpp
OMP_SRCS := OpenMP.cpp log.cpp

OBJ_DIR := build
SEQ_OBJ_DIR := $(OBJ_DIR)/seq
OMP_OBJ_DIR := $(OBJ_DIR)/omp

SEQ_OBJS := $(addprefix $(SEQ_OBJ_DIR)/,$(SEQ_SRCS:.cpp=.o))
OMP_OBJS := $(addprefix $(OMP_OBJ_DIR)/,$(OMP_SRCS:.cpp=.o))

.PHONY: all clean

all: $(SEQ_TARGET) $(OMP_TARGET)

$(SEQ_TARGET): $(SEQ_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OMP_TARGET): $(OMP_OBJS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(SEQ_OBJ_DIR)/%.o: %.cpp | $(SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OMP_OBJ_DIR)/%.o: %.cpp | $(OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(SEQ_OBJ_DIR):
	@mkdir -p $@

$(OMP_OBJ_DIR):
	@mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(SEQ_TARGET) $(OMP_TARGET) output

