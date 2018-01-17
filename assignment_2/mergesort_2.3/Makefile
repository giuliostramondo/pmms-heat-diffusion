
PRJ=mergesort
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))

CC=gcc

ifndef DEBUG
CFLAGS= -std=c99 -O2
LIB=
else
CFLAGS= -std=c99 -O2 -g3
LIB=
endif

all: $(PRJ)

$(PRJ): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LIB)	

clean:
	-rm -f $(OBJ) $(PRJ)
