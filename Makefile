STUDENTID1=<HERE ID OF THE STUDENT>
STUDENTID2=<HERE ID OF THE STUDENT>
GROUP=<HERE GROUP ID>

HEADERS=$(wildcard include/*.h)
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))

ASSIGNMENTS=pth omp seq
TARGETS=$(addprefix heat-,$(ASSIGNMENTS))

ASSIGNMENTS_SRC=$(wildcard $(addsuffix /*.c, $(ASSIGNMENTS)))
ASSIGNMENTS_OBJ=$(subst /,/obj/,$(patsubst %.c,%.o,$(ASSIGNMENTS_SRC)))

CC=gcc
CFLAGS=-I./include

define filter-assignment-obj 
	$(filter $(1)/%,$(ASSIGNMENTS_OBJ))
endef


#This expands into rules to compile all the assignments
all: $(TARGETS)
	@echo "headers :$(HEADERS)\nsrc: $(SRC)\nobj: $(OBJ)"
	@echo "ASSIGNMENTS_SRC :$(ASSIGNMENTS_SRC)\nASSIGNMENTS_OBJ:$(ASSIGNMENTS_OBJ)"
	@echo $(addprefix heat-,$(ASSIGNMENTS))


#This generates rules for each assignment
.SECONDEXPANSION:
$(TARGETS): heat-%:  $$(call filter-assignment-obj, %) $(OBJ) $(HEADERS)  	
	@echo "Compiling $@ in $(patsubst heat-%,%,$@) which depends from $+ "
	$(CC) $(CFLAGS) $< -o $@	

#This generates rules for each object of an assignment 
.SECONDEXPANSION:
$(ASSIGNMENTS_OBJ): %.o :$$(subst /obj/,/,%.c) 
	@echo "Compiling assignment obj $@: gcc -c $< -o $@"
	if [ ! -d "$(dir $(<))obj" ]; then mkdir $(dir $(<))obj; fi
	$(CC) $(CFLAGS) -c $< -o $@	

#Compiling framework objects and putting them in the obj folder
obj/%.o: src/%.c $(HEADERS)
	@echo "Compiling framework $@ with: gcc -c $< -o $@"
	if [ ! -d "obj" ]; then mkdir obj; fi
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	-rm -rf obj
	-rm -rf $(addsuffix /obj,$(ASSIGNMENTS))
	-rm -f $(TARGETS)
