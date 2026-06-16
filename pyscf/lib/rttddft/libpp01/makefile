.SUFFIXES:
.SUFFIXES: .f90 .f .o .c
.f.o:
	gfortran -c -Wall $<
.f90.o:
	gfortran -c -Wall $<
.c.o  :
	gcc -fPIC -Wall -lm -c $<



libpp01:xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o govlp.o cstrutil.o Properties.o
	gcc -Wall -shared -o $@.so $^ $(GOPT) -lm -llapack -lblas -lgfortran

test2:xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o govlp.o cstrutil.o Properties.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran
	ls -ltrh $@.x

MAIN_OBJ=Gaussian.o xgaussope.o testmain.o dalloc.o mathfn.o cmplx.o
test_main: $(MAIN_OBJ)
	gcc -Wall -Wall -o $@.x   $^ $(GOPT) -lm -llapack -lblas -lgfortran
	ls -ltrh $@.x
.PHONY: clean
clean:
	@rm -f *.o *.mod

#testlib:zgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o govlp.o cstrutil.o Properties.o
#	gcc -Wall -shared -o $@.so $^ $(GOPT) -lm -llapack -lblas -lgfortran

testread:cstrutil.o xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o govlp.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran

libctest:xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o
	gcc -Wall -shared -o $@.so $^ $(GOPT) -lm -llapack -lblas -lgfortran

testmain:testmain01.o xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran

test_alloc:test_alloc.o xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran


test_govlp:govlp.o xgaussope.o Gaussian.o dalloc.o mathfn.o cmplx.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran

testlib:testlib.o
	gcc -Wall -shared -o $@.so $^ $(GOPT) -lm -llapack -lblas -lgfortran

test_Gaussian:test_Gaussian.o Gaussian.o xgaussope.o dalloc.o cstrutil.o mathfn.o govlp.o cmplx.o
	gcc -Wall -o $@.x $^ $(GOPT) -lm -llapack -lblas -lgfortran
