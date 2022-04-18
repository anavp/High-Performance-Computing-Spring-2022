
CXX = nvcc
CXXFLAGS = -Xcompiler -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	$(RM) -f $(TARGETS) *~

.PHONY: all, clean