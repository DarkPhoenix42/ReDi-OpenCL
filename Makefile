# Makefile for ReDi-OpenCL

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Ofast -Wall -Wextra -Wpedantic

# Include directories
INCLUDES = -Isrc -Iinclude

# Libraries
LDLIBS = -lSDL2 -lOpenCL

# Source files
SRCS = src/ReDi.cpp

# Executable name
EXEC = build/ReDi

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC):
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDLIBS) $(INCLUDES) -o $(EXEC)

# Clean
clean:
	rm -f $(OBJS) $(EXEC)

# Install dependencies
install: install_sdl2 install_opencl install_json install_argparse

install_sdl2:
	@echo "Please install SDL2 developer libraries."
	@echo "For Ubuntu/Debian: sudo apt-get install libsdl2-dev"
	@echo "For Fedora: sudo dnf install SDL2-devel"
	@echo "For Arch: sudo pacman -S sdl2"

install_opencl:
	@echo "Please install OpenCL runtime and headers."
	@echo "For Ubuntu/Debian: sudo apt-get install ocl-icd-opencl-dev"
	@echo "For Fedora: sudo dnf install opencl-headers"
	@echo "For Arch: sudo pacman -S opencl-headers"

install_json:
	@echo "Downloading nlohmann/json.hpp..."
	mkdir -p include
	curl -L https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp -o include/json.hpp

install_argparse:
	@echo "Downloading argparse.hpp..."
	mkdir -p include
	curl -L https://raw.githubusercontent.com/p-ranav/argparse/master/include/argparse/argparse.hpp -o include/argparse.hpp

# Phony targets
.PHONY: all clean install install_sdl2 install_opencl install_json install_argparse