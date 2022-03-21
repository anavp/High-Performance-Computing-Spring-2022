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
	$(CC) -o val_test01_solved ./val_test01.cpp $(LIBS)
	$(CC) -o val_test02_solved ./val_test02.cpp $(LIBS)
	$(CC) -o omp_solved2 ./omp_solved2.cpp $(LIBS)
	$(CC) -o omp_solved3 ./omp_solved3.cpp $(LIBS)
	$(CC) -o omp_solved4 ./omp_solved4.cpp $(LIBS)
	$(CC) -o omp_solved5 ./omp_solved5.cpp $(LIBS)
	$(CC) -o omp_solved6 ./omp_solved6.cpp $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 
	rm -f linker
