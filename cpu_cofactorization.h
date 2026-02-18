#include "gmp.h"
#include <stdint.h>

int cofactorisation(int first_side, mpz_t* large_primes1, mpz_t* large_primes2, 
	mpz_t* large_factors, uint16_t* lpb, uint32_t* nlp, int only_mpqs,
	int* num_mpqs, int* num_mpqs3, int* num_ecm64, int* num_ecm128);