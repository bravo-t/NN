CC=gcc
LIBS=-lm
CFLAGS=-g -std=c99 -Wall

test: test.c matrix_operations.o 
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

matrix_operations.o: util/matrix_operations.c util/matrix_type.h util/matrix_operations.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

.PHONY: clean
clean:
	rm -f *.o
