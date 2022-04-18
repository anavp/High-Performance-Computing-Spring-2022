
CXX = nvcc
CXXFLAGS = -Xcompiler -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	$(RM) $(TARGETS) *~

.PHONY: all, clean