#Makefile

CC :=g++

FLAG := -fopenmp -O3 -lm

SOURCE := \
BVD_code_5eq.cpp \

OBJDIR := ./obj/

OBJ := $(SOURCE:%.cpp=$(OBJDIR)%.o)

INCLUDE := header.hpp

Main : $(OBJ)
	$(CC) -o run.out $(OBJ) $(FLAG)

debug : $(SOURCE)
	$(CC) -o dbg.out $(SOURCE) -fopenmp -O0 -g3 -ggdb3 -lm

$(OBJDIR)%.o : %.cpp
	@[ -d $(OBJDIR) ]
	$(CC) -o $@ -c $< $(FLAG)

clean :
	rm -f $(OBJDIR)*.o *.out
