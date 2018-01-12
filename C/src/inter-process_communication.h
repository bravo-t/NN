#ifndef __IPC_HEADER__
#define __IPC_HEADER__

#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <semaphore.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>

void microsecSleep (long ms);
void waitUntilEveryoneIsFinished(sem_t *sem);
int IPCWriteToSharedMem(char* shared_memory_name, void* base_ptr, int data_length);
int IPCReadFromSharedMem(char* shared_memory_name, void* data, int data_length);
int IPCRemoveSharedMemFile(char* shared_memory_name);

#endif