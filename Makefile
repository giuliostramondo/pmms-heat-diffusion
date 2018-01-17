#MODIFY the following variables (STUDENTID1, STUDENTID2, GROUP) with your infos
STUDENTID1=HERE_ID_OF_THE_STUDENT
STUDENTID2=HERE_ID_OF_THE_STUDENT
GROUP=HERE_GROUP_ID



# This Makefile expects the directories defined in the ASSIGNMENTS variable to be in 
# its directory. An executable binary will be generated for each directory in the 
# ASSIGNMENTS variable; this will be named heat-<dirname>.
# Every ".c" source file in one of the assignment directory will be compiled separately
# and the generated ".o" file will be saved in <assignment_dir>/obj.
# Each executable is then generated linking the framework common objects, to the objects
# of contained in <assignment_dir>/obj.
# The framework common objects are generated in the top level ./obj directory, their 
# relative src files are located in the ./src folder

#To add an additional target that has to be compiled with the framework, create a folder
# <target_name> and place the relative ".c" source files inside it. You can specify
# specific compiler flags for this target setting the CFLAGS_<target_name> variable.
#After make the source files in the <target_name> folder will be compiled and linked with
#the rest of the framework, producing an heat-<target_name> executable.

#To add an additional target (compiled independently from the framework) to this makefile, 
#append the target name to the TARGETS variable and add a new rule. You can use omp-merge 
#as an example.

HEADERS=$(wildcard include/*.h)
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))


ASSIGNMENTS=pth omp seq #acc
TARGETS=$(addprefix heat-,$(ASSIGNMENTS)) omp-merge

ASSIGNMENTS_SRC=$(wildcard $(addsuffix /*.c, $(ASSIGNMENTS)))
ASSIGNMENTS_OBJ=$(subst /,/obj/,$(patsubst %.c,%.o,$(ASSIGNMENTS_SRC)))

#Specify the compiler to use
CC=gcc
#Specify the global compiler flags used for the framework and EVERY assignment
CFLAGS=-I./include -std=c99
LIBS=

#Compiler flags used for the seq code
CFLAGS_seq=-g3
LIBS_seq=

#Compiler flags used for the omp code
CFLAGS_omp=-O2
LIBS_omp=
#Compiler flags used for the pth code
CFLAGS_pth=-O2
LIBS_pth=-lm -pthread 
#Compiler flags used for the acc code
CFLAGS_acc=-g3 -fopenacc
LIBS_acc=



define filter-assignment-obj 
	$(filter $(1)/%,$(ASSIGNMENTS_OBJ))
endef


#This expands into rules to compile all the assignments
all: $(TARGETS)


omp-merge: omp_merge/merge.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

#This generates rules for each assignment
.SECONDEXPANSION:
$(filter heat-%,$(TARGETS)): heat-%:  $$(call filter-assignment-obj, %) $(OBJ) $(HEADERS)  	
	@echo "Compiling $@ in $(patsubst heat-%,%,$@) which depends from $+ "
	$(CC) $(CFLAGS) $(CFLAGS_$(patsubst heat-%,%,$@)) $(filter %.o,$+) -o $@ $(LIBS) $(LIBS_$(patsubst heat-%,%,$@))



#This generates rules for each object of an assignment 
.SECONDEXPANSION:
$(ASSIGNMENTS_OBJ): %.o :$$(subst /obj/,/,%.c) 
	@echo "Compiling assignment $(patsubst %/,%, $(dir $<)) file $@:"
	@if [ ! -d "$(dir $(<))obj" ]; then mkdir $(dir $(<))obj; fi
	$(CC) $(CFLAGS) $(CFLAGS_$(patsubst %/,%, $(dir $<))) -c $< -o $@ $(LIBS) $(LIBS_$(patsubst %/,%, $(dir $<)))

#Compiling framework objects and putting them in the obj folder
obj/%.o: src/%.c $(HEADERS)
	@echo "Compiling framework file $@: gcc -c $< -o $@"
	@if [ ! -d "obj" ]; then mkdir obj; fi
	$(CC) $(CFLAGS) -c $< -o $@ $(LIBS)


.PHONY: clean submission demo

demo: $(basename $(filter %.c,$(wildcard demo/*)) )

$(basename $(filter %.c,$(wildcard demo/*)) ): $(filter-out obj/main.o, $(OBJ))
	gcc $(CFLAGS) $@.c $(filter-out obj/main.o, $(OBJ)) -o $@ $(LIBS)

submission:
	$(eval BASEDIR := $(notdir $(shell pwd)))
	touch heat_submission_group_$(GROUP)_$(STUDENTID1)_$(STUDENTID2).tar.gz 
	cd ..;tar --exclude=$(BASEDIR)/heat_submission_group_$(GROUP)_$(STUDENTID1)_$(STUDENTID2).tar.gz -czf $(BASEDIR)/heat_submission_group_$(GROUP)_$(STUDENTID1)_$(STUDENTID2).tar.gz $(BASEDIR)

clean:
	-rm -rf obj
	-rm -rf $(addsuffix /obj,$(ASSIGNMENTS))
	-rm -f $(TARGETS)
	-rm -f demo/main_imgdemo    demo/main_inputdemo    demo/main_outputdemo 
