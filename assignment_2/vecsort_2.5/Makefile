
PRJ=vecsort
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))

CC=gcc

ifndef DEBUG
CFLAGS=-O2 -std=c99
LIB=
else
CFLAGS=-O2 -g3 -std=c99
LIB=
endif

all: $(PRJ)

$(PRJ): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LIB)	

clean:
	-rm -f $(OBJ) $(PRJ)
