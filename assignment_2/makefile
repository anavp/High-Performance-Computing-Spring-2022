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
	$(CC) -o val_test01 ./valgrind_test/val_test01.cpp $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 
	rm -f linker
