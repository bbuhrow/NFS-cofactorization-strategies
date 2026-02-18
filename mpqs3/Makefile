# Copyright (C) 2001,2002 Jens Franke
# This file is part of gnfs4linux, distributed under the terms of the 
# GNU General Public Licence and WITHOUT ANY WARRANTY.

# You should have received a copy of the GNU General Public License along
# with this program; see the file COPYING.  If not, write to the Free
# Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.


#gcc on osx
#CFLAGS= -m64 -Ofast -march=native -I/users/buhrow/src/c/gmp_install/gmp-6.2.0/include -Wa,-q
#CFLAGS= -m64 -Ofast -march=native -I/usr/local/Cellar/gmp4/4.3.2/include -Wa,-q

#windows
#CFLAGS= -Wall  -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable  -Ofast -fomit-frame-pointer -march=corei7 -mtune=corei7 -I. -Ic:/users/steve/progs/local/include
#CFLAGS= -Wimplicit-function-declaration -m64 -Ofast -march=native -Ic:/users/steve/progs/local/include

#clang
#CFLAGS= -Wall -Wno-unused-variable -Wno-unused-function -m64 -Ofast -march=native -I/Users/searle/progs/ensc-dependencies/include
#CFLAGS= -m64 -Ofast -march=corei7 -I/Users/searle/progs/ensc-dependencies/include 
#clang for profiling
#CFLAGS= -O1 -g -march=native -I/Users/searle/progs/ensc-dependencies/include

#linux
CFLAGS= -m64 -g -Ofast -march=native -I. -I../../gmp-install/6.2.0-gcc/include -fcommon

CC=gcc $(CFLAGS)
#CC=gcc-4.9 $(CFLAGS)
#CC=clang $(CFLAGS)


SRCFILES=mpqs-config.h mpqs3_sieve.asm mpqs_eval.asm mpqs_gauss.asm if.c mpqs.c mpqs3.c \
	mpqs_nextpol.asm mpqs_sieve.asm mpqs_sieveinit.asm mpqs_td.asm \
	mpqs3arith.c mpqs3_nextpol.asm mpqs3_td.asm gmp-aux.c \
	mpqs3_gauss.asm psp.c asm_zeit.asm zeit.c montgomery_mul.c \
	asm_arith64.asm asm_arith128.asm asm_arith192.asm pt64.asm asm_64bit.asm \
	modinv32.c modinv1000.asm modinv1000b.asm basemath.c noasm64.c invtab.c


%.S: %.asm
	m4 -g  -Dn_i_bits=0 ls-defs.asm $^ > $@

%.o: %.S
	$(CC) -c $^

%.o: %.s
	$(CC) -c $^

.SECONDARY:  *.s *.S

.PHONY: bup

all: liblasieve.a 

mpqs.o: siever-config.h mpqs.h
mpqs3.o: siever-config.h mpqs3.h

liblasieve.a: psp.o if.o mpqs.o mpqs3.o mpqs_td.o mpqs_sieve.o mpqs_eval.o mpqs_sieveinit.o \
	mpqs_nextpol.o mpqs_gauss.o mpqs3_sieve.o mpqs3_nextpol.o mpqs3_td.o \
	zeit.o asm-zeit.o mpqs3_gauss.o montgomery_mul.o gmp-aux.o \
	asm_arith64.o asm_arith128.o asm_arith192.o pt64.o asm_64bit.o \
	modinv32.o modinv1000.o modinv1000b.o basemath.o noasm64.o invtab.o
	ar rcs $@ $^

bup:
	echo $(SRCFILES)

clean:
	rm *.o *.a
