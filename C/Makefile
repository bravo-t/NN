ifeq ($(DEBUG),1)
	PRE_CFLAGS = -DDEBUG -g -O0
else
	PRE_CFLAGS = -g -O0
endif
CC=gcc
LIBS=-lm
CFLAGS=$(PRE_CFLAGS) -std=c99 -Wall

default: test

test: test.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

misc_utils.o: util/misc_utils.c util/misc_utils.h util/matrix_type.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

matrix_operations.o: util/matrix_operations.c util/matrix_operations.h misc_utils.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

layers.o: util/layers.c util/layers.h matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

fully_connected_net.o: util/fully_connected_net.c util/fully_connected_net.h layers.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)


.PHONY: clean
clean:
	rm -f *.o *~ util/*~ test
