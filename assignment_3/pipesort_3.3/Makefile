
PRJ=pipesort
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))

CC=gcc

ifndef DEBUG
CFLAGS=-O2
LIB=
else
CFLAGS=-O2 -g3
LIB=
endif

all: $(PRJ)

$(PRJ): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LIB)	

clean:
	-rm -f $(OBJ) $(PRJ)
