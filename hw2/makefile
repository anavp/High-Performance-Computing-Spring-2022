IDIR =./
CC=g++
CFLAGS=-I$(IDIR)

ODIR=./

LIBS=-lm

# _DEPS = linker.h
# DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# _OBJ = linker.o
# OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp# $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


hw2: #$(OBJ)
	$(CC) -o val_test01_solved ./val_test01_solved.cpp $(LIBS)
	$(CC) -o val_test02_solved ./val_test02_solved.cpp $(LIBS)
	$(CC) -o omp_solved2 -fopenmp ./omp_solved2.c $(LIBS)
	$(CC) -o omp_solved3 -fopenmp ./omp_solved3.c $(LIBS)
	$(CC) -o omp_solved4 -fopenmp ./omp_solved4.c $(LIBS)
	$(CC) -o omp_solved5 -fopenmp ./omp_solved5.c $(LIBS)
	$(CC) -o omp_solved6 -fopenmp ./omp_solved6.c $(LIBS)
	$(CC) -o MMult1 -fopenmp -march=native -O3 ./MMult1.cpp $(LIBS)
	$(CC) -o jacobi2D-omp -fopenmp ./jacobi2D-omp.cpp $(LIBS)
	$(CC) -o gs2D-omp -fopenmp ./gs2D-omp.cpp $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 
	rm -f linker
