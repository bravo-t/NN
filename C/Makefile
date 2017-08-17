ifeq ($(DEBUG),1)
	PRE_CFLAGS = -DDEBUG -g -O0
else
	PRE_CFLAGS = -g -O0
endif
CC=gcc
LIBS=-lm
CFLAGS=$(PRE_CFLAGS) -std=gnu99 -Wall

default: fully_connected_network

fully_connected_network: fully_connected_network.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

misc_utils.o: src/misc_utils.c src/misc_utils.h src/matrix_type.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

matrix_operations.o: src/matrix_operations.c src/matrix_operations.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

layers.o: src/layers.c src/layers.h matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

fully_connected_net.o: src/fully_connected_net.c src/fully_connected_net.h layers.o matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

unit_test: unit_test.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

.PHONY: clean
clean:
	rm -f *.o *~ src/*~ test
