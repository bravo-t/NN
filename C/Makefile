ifdef DEBUG
	PRE_CFLAGS = -DDEBUG=$(DEBUG) -g -O0
else
	PRE_CFLAGS = -g -O0
endif
CC=gcc
LIBS=-lm
CFLAGS=$(PRE_CFLAGS) -std=gnu99 -Wall

default: fully_connected_network convolutional_network 

fully_connected_network: fully_connected_network.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

fully_connected_network_multithread: fully_connected_network_multithread.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o matrix_operations_multithread.o layers_multithread.o fully_connected_net_multithread.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

misc_utils.o: src/misc_utils.c src/misc_utils.h src/network_type.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

matrix_operations.o: src/matrix_operations.c src/matrix_operations.h
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

matrix_operations_multithread.o: src/matrix_operations_multithread.c src/matrix_operations_multithread.h matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

layers.o: src/layers.c src/layers.h matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

layers_multithread.o: src/layers_multithread.c src/layers_multithread.h matrix_operations.o matrix_operations_multithread.o layers.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

fully_connected_net.o: src/fully_connected_net.c src/fully_connected_net.h layers.o matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

fully_connected_net_multithread.o: src/fully_connected_net_multithread.c src/fully_connected_net_multithread.h matrix_operations.o layers.o layers_multithread.o matrix_operations_multithread.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

unit_test: unit_test.c misc_utils.o matrix_operations.o layers.o fully_connected_net.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

convnet_operations.o: src/convnet_operations.c src/convnet_operations.h matrix_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

convnet_layers.o: src/convnet_layers.c src/convnet_layers.h convnet_operations.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

convnet.o: src/convnet.c src/convnet.h convnet_layers.o
	$(CC) -c -o $@ $< $(LIBS) $(CFLAGS)

convnet_test: convnet_test.c convnet.o convnet_layers.o convnet_operations.o matrix_operations.o layers.o fully_connected_net.o misc_utils.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

convolutional_network: convolutional_network.c convnet.o convnet_layers.o convnet_operations.o matrix_operations.o layers.o fully_connected_net.o misc_utils.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

mnist_test: mnist.c convnet.o convnet_layers.o convnet_operations.o matrix_operations.o layers.o fully_connected_net.o misc_utils.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

convnet_unit_test: convnet_unit_test.c convnet.o convnet_layers.o convnet_operations.o matrix_operations.o layers.o fully_connected_net.o misc_utils.o
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS)

.PHONY: clean
clean:
	rm -f *.o *~ src/*~ 
