/*1:*/
#ifndef __SIEVER_CONFIG_H__
#define __SIEVER_CONFIG_H__
#include <sys/types.h> 
#include <gmp.h> 




#define HAVE_CMOV

#ifdef _WIN64
#define ASM_ATTR   __attribute__((sysv_abi))
#else
#define ASM_ATTR
#endif

#ifdef _WIN64
void bzero(void*,size_t);
#endif

int psp(mpz_t n);

#define L1_BITS 15
#define ULONG_RI
typedef unsigned u32_t;
typedef int i32_t;
typedef short int i16_t;
typedef unsigned short u16_t;


#ifdef _WIN64
typedef unsigned long long u64_t;
typedef long long i64_t;
typedef unsigned short ushort;
typedef unsigned long long ulong;
#else
 typedef unsigned long u64_t;
typedef long i64_t;
typedef unsigned short ushort;
typedef unsigned long ulong;
#endif

#define U32_MAX 0xffffffff
#define I32_MAX INT_MAX

int asm_cmp(ulong*a,ulong*b);

#define  MAX_FB_PER_P 2
#define ASM_RESCALE

#if 1
#define VERY_LARGE_Q
#endif

void ASM_ATTR rescale_interval1(unsigned char*,u64_t);
void ASM_ATTR rescale_interval2(unsigned char*,u64_t);

u64_t ASM_ATTR asm_modadd64(u64_t,u64_t);
u64_t ASM_ATTR asm_modmul64(u64_t,u64_t);

#define FB_RAS 3
#define N_PRIMEBOUNDS 12

#endif

