# ************ CCC - passing optimization flags ************ 
ifeq ($(CCC_OPTS),)
 CCC_OPTS = -newgvn -rpo-functionattrs -mergeicmps -mem2reg -gvn-hoist
endif
ifeq ($(CCC_OPTS_ADD),)
 CCC_OPTS_ADD =
endif

# Declarations
ifeq ($(ZCC),) 
 ZCC = /home/ryanl/cse583-heureka/LLVM_DIR/install/bin/clang
endif
ifeq ($(CC_OPTS),)
 CC_OPTS = -emit-llvm -c
endif
ifeq ($(LDCC),)
 LDCC = /home/ryanl/cse583-heureka/LLVM_DIR/install/bin/llvm-link
endif
ifeq ($(LD_OPTS),)
 LD_OPTS = -o linked.bc
endif
OPT = /home/ryanl/cse583-heureka/LLVM_DIR/install/bin/opt
OPT_OPTS = $(CCC_OPTS)
# Actions
all:		
	$(ZCC) $(CC_OPTS) *.c
	$(LDCC) *.bc $(LD_OPTS)
	$(OPT) -p $(OPT_OPTS) linked.bc -o optimized.bc
	$(ZCC) optimized.bc -o a.out -lm
clean:
	rm -f a.out *.o *.a *.s *.i *.I *.bc
