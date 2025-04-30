# ************ CCC - passing optimization flags ************ 
ifeq ($(CCC_OPTS),)
 CCC_OPTS = -gvn -newgvn -attributor -mem2reg -jump-threading
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

# Actions
all:		
	$(ZCC) $(CC_OPTS) -Oz -DSASR -DSTUPID_COMPILER -DNeedFunctionPrototypes=1 *.c 
	$(LDCC) *.bc $(LD_OPTS)
	$(OPT) $(CCC_OPTS) linked.bc -o optimized.bc
	$(ZCC) -Oz optimized.bc -o a.out -lm
clean:
	rm -f a.out *.o *.a *.s *.i *.I *.bc

# $(ZCC) $(CCC_OPTS) $(CCC_OPTS_ADD) -c -DSASR -DSTUPID_COMPILER -DNeedFunctionPrototypes=1 *.c
# $(LDCC) $(LD_OPTS) $(CCC_OPTS_ADD) -lm *.o
