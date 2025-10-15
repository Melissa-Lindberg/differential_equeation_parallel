CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS :=
TARGET := consistent
SRCS := consistent.cpp log.cpp
OBJ_DIR := build
OBJS := $(SRCS:%.cpp=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
