#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gmp.h"
#include "microecm.h"
#include "tinyecm.h"
#include "mpqs3/mpqs.h"
#include "mpqs3/mpqs3.h"
#include "mpqs3/if.h"
#include "mpqs3/gmp-aux.h"

#if 1

static uint64_t pran;
static mpz_t uecm_factors[3];
static int uecm_initialized = 0;

int cofactorisation(int first_side, mpz_t* large_primes1, mpz_t* large_primes2,
	mpz_t* large_factors, uint16_t* lpb, uint32_t* nlp, int only_mpqs,
	int *num_mpqs, int *num_mpqs3, int *num_ecm64, int *num_ecm128)
{
	uint32_t s, nb[2];
	uint32_t m, * fm, t, j, done[2], B1, B2, pm1done[2];
	clock_t cl;
	mpz_t* large_primes;


	if (!uecm_initialized)
	{
		// for tinyecm
		mpz_init(uecm_factors[0]);
		mpz_init(uecm_factors[1]);
		mpz_init(uecm_factors[2]);
		pran = 42;
		uecm_initialized = 1;
	}

	for (s = 0; s < 2; s++) {
		if (mpz_sgn(large_factors[s]) > 0) {
			if (mpz_cmp_ui(large_factors[s], 1) == 0)
				nlp[s] = 0;
			else {
				nlp[s] = 1;
				if (s == 0)
				{
					mpz_set(large_primes1[0], large_factors[s]);
				}
				else
				{
					mpz_set(large_primes2[0], large_factors[s]);
				}
			}
			nb[s] = 0;
		}
		else {
			mpz_neg(large_factors[s], large_factors[s]);
			nb[s] = (uint32_t)(mpz_sizeinbase(large_factors[s], 2));
			nlp[s] = 2;
		}
	}

	if ((nlp[0] < 2) && (nlp[1] < 2)) {
		return 0;
	}

	for (s = 0; s < 2; s++) {
		if (nlp[s] == 2) { nlp[s] = 0; done[s] = 0; pm1done[s] = 0; }
		else done[s] = 2;
	}

	for (s = first_side;  ; s = 0 - s + 1) {
		int32_t nf, i;
		size_t sf[2];
		mpz_t* fac;

		//gmp_printf("running side %u on inputs %Zd and %Zd\n", s, large_factors[0], large_factors[1]);

		if (done[s] > 1)
		{
			if (s != first_side)
				break;
			else
				continue;
		}

		if (only_mpqs)
		{
			if (mpz_sizeinbase(large_factors[s], 2) > 96)
			{
				nf = mpqs3_factor(large_factors[s], lpb[s], &fac);
				*num_mpqs3 += 1;

				if ((nf < 3) && (mpz_sizeinbase(large_factors[s], 2) > (lpb[s] * 2)))
				{
					printf("%Zd doesn't completely factor: ", large_factors[s]);
					for (i = 0; i < nf; i++)
						gmp_printf("%Zd ", fac[i]);
					printf("\n");
					nf = 0;
				}
					
			}
			else
			{
				nf = mpqs_factor(large_factors[s], lpb[s], &fac);
				*num_mpqs += 1;

				if ((nf < 3) && (mpz_sizeinbase(large_factors[s], 2) > (lpb[s] * 2)))
				{
					printf("%Zd doesn't completely factor: ", large_factors[s]);
					for (i = 0; i < nf; i++)
						gmp_printf("%Zd ", fac[i]);
					printf("\n");
					nf = 0;
				}
			}
		}
		else
		{

			nf = 0;
			fac = uecm_factors;

			if (mpz_sizeinbase(large_factors[s], 2) <= 64) {

				uint64_t n64 = large_factors[s]->_mp_d[0]; // mpz_get_ull(large_factors[s]);
				uint64_t f = getfactor_uecm(n64, 0, &pran);
				*num_ecm64 += 1;
				if (f > 1)
				{
					mpz_set_ull(fac[0], f);
					mpz_tdiv_q_ull(fac[1], large_factors[s], f);

					nf = 2;

					if (mpz_sizeinbase(fac[0], 2) > lpb[s])
					{
						nf = 0;
					}
					if (mpz_sizeinbase(fac[1], 2) > lpb[s])
					{
						nf = 0;
					}
					if (mpz_probab_prime_p(fac[0], 1) == 0)
					{
						gmp_printf("ecm64 found a composite factor %Zd of %Zd (1a)\n", 
							fac[0], large_factors[s]);
						nf = 0;
					}
					if (mpz_probab_prime_p(fac[1], 1) == 0)
					{
						gmp_printf("residue after ecm64 is composite %Zd of %Zd (2a)\n", 
							fac[1], large_factors[s]);
						nf = 0;
					}
				}
				else
				{
					// uecm failed, which does sometimes happen
					nf = mpqs_factor(large_factors[s], lpb[s], &fac);
					*num_mpqs += 1;

					for (i = 0; i < nf; i++)
					{
						if (mpz_sizeinbase(fac[i], 2) > lpb[s])
							nf = 0;
					}
				}
			}
			else
			{

#if defined(USE_AVX512F)

				// it is only faster to use ecm on large inputs
				// if we have the parallel 104-bit implementation
				// in getfactor_tecm_x8

				if (mpz_sizeinbase(large_factors[s], 2) > 104)
				{
					nf = mpqs3_factor(large_factors[s], lpb[s], &fac);
					*num_mpqs3 += 1;

					for (i = 0; i < nf; i++)
					{
						if (mpz_sizeinbase(fac[i], 2) > lpb[s])
							nf = 0;
					}
				}
				else if (getfactor_tecm_x8(large_factors[s], fac[0],
					mpz_sizeinbase(large_factors[s], 2) / 3 - 2, &pran) > 0)
				{
					*num_ecm128 += 1;

					if (mpz_sizeinbase(fac[0], 2) <= lpb[s])
					{
						mpz_tdiv_q(fac[1], large_factors[s], fac[0]);

						// if the remaining residue is obviously too big, we're done.
						if (mpz_sizeinbase(fac[1], 2) > ((lpb[s] * 2)))
						{
							nf = 0;
							goto done;
						}

						// check if the residue is prime.  could again use
						// a cheaper method.
						if (mpz_probab_prime_p(fac[1], 1) > 0)
						{
							if (mpz_sizeinbase(fac[1], 2) <= lpb[s])
							{
								// we just completed a DLP factorization involving
								// 2 primes whos product was > 64 bits.
								nf = 2;
								goto done;
							}
							nf = 0;
							goto done;
						}

						// ok, so we have extracted one suitable factor, and the 
						// cofactor is not prime and a suitable size.  Do more work to 
						// split the cofactor.
						// todo: target this better based on expected factor size.
						uint64_t q64;
						uint64_t f64;
						if (mpz_sizeinbase(fac[1], 2) <= 64)
						{
							q64 = mpz_get_ull(fac[1]);
							f64 = getfactor_uecm(q64, 0, &pran);
							mpz_set_ull(fac[2], f64);
							*num_ecm64 += 1;
						}
						else
						{
							// we have a composite residue > 64 bits.  
							// use ecm first with high effort.
							//getfactor_tecm(fac[1], fac[2], 32, &pran);
							// *num_ecm128 += 1;
							f64 = 1;
						}

						if (f64 > 1)
						{
							mpz_tdiv_q_ull(fac[1], fac[1], f64);
							nf = 3;

							if (mpz_sizeinbase(fac[1], 2) > lpb[s]) {
								nf = 0;
							}
							if (mpz_sizeinbase(fac[2], 2) > lpb[s]) {
								nf = 0;
							}
							if (mpz_probab_prime_p(fac[0], 1) == 0)
							{
								nf = 0;
							}
							if (mpz_probab_prime_p(fac[1], 1) == 0)
							{
								nf = 0;
							}
							if (mpz_probab_prime_p(fac[2], 1) == 0)
							{
								nf = 0;
							}
						}
						else
						{
							// uecm/tecm failed or input was too large
							nf = mpqs_factor(fac[1], lpb[s], &fac);
							*num_mpqs += 1;

							for (i = 0; i < nf; i++)
							{
								if (mpz_sizeinbase(fac[i], 2) > lpb[s])
									nf = 0;
							}

							if (nf == 2)
							{
								// fac is now set to mpqs's statically allocated
								// set of mpz_t's.  copy in the one we found by ecm.
								nf = 3;
								mpz_set(fac[2], uecm_factors[0]);
							}
							else
							{
								nf = 0;
							}
						}
					}
					else
					{
						// found a factor larger than the lpb.
						// check if the factor is prime.  could again use
						// a cheaper method.
						if (mpz_probab_prime_p(fac[0], 1) > 0)
						{
							// if the factor is obviously too big, give up.  This isn't a
							// failure since we haven't expended much effort yet.
							nf = 0;
						}
						else
						{
							// tecm found a composite first factor.
							// if it is obviously too big, we're done.
							if (mpz_sizeinbase(fac[0], 2) > ((lpb[s] * 2)))
							{
								nf = 0;
								goto done;
							}

							// isolate the 2nd smaller factor, and check its size.
							mpz_tdiv_q(fac[1], large_factors[s], fac[0]);

							if (mpz_sizeinbase(fac[1], 2) > (lpb[s]))
							{
								nf = 0;
								goto done;
							}

							// todo: target this better based on expected factor size.
							uint64_t q64;
							uint64_t f64;
							if (mpz_sizeinbase(fac[0], 2) <= 64)
							{
								q64 = mpz_get_ull(fac[0]);
								f64 = getfactor_uecm(q64, 0, &pran);
								mpz_set_ull(fac[2], f64);
								*num_ecm64 += 1;
							}
							else
							{
								// split with mpqs below
								f64 = 1;
							}

							if (f64 > 1)
							{
								mpz_tdiv_q_ull(fac[0], fac[0], f64);
								nf = 3;

								if (mpz_sizeinbase(fac[0], 2) > lpb[s]) {
									nf = 0;
								}
								if (mpz_sizeinbase(fac[2], 2) > lpb[s]) {
									nf = 0;
								}
								if (mpz_probab_prime_p(fac[0], 1) == 0)
								{
									nf = 0;
								}
								if (mpz_probab_prime_p(fac[1], 1) == 0)
								{
									nf = 0;
								}
								if (mpz_probab_prime_p(fac[2], 1) == 0)
								{
									nf = 0;
								}

							}
							else
							{
								// uecm failed or input was too large
								nf = mpqs_factor(fac[0], lpb[s], &fac);
								*num_mpqs += 1;

								if (nf == 2)
								{
									// fac is now set to mpqs's statically allocated
									// set of mpz_t's.  copy in the one we found by ecm.
									nf = 3;
									mpz_set(fac[2], uecm_factors[1]);
								}
								else
								{
									nf = 0;
								}
							}
						}
					}
				}
				else
				{
					// if ecm can't find a factor, give up.  
					// unless this is a DLP with lpbr/a > 32... i.e., if the
					// large factor size is greater than 64 bits but less than
					// lpbr/a * 2.  In that case run mpqs... or tecm with
					// greater effort.
					*num_ecm128 += 1;

#if 0
					if (mpz_sizeinbase(large_factors[s1], 2) <= (lpb[s1] * 2))
					{
						if (getfactor_tecm(large_factors[s1], factor1, 33, &pran) > 0)
						{
							if (mpz_sizeinbase(factor1, 2) <= lpb[s1])
							{
								mpz_tdiv_q(factor2, large_factors[s1], factor1);

								// check if the residue is prime.  could again use
								// a cheaper method.
								if (mpz_probab_prime_p(factor2, 1) > 0)
								{
									if (mpz_sizeinbase(factor2, 2) <= lpb[s1])
									{
										// we just completed a DLP factorization involving
										// 2 primes whos product was > 64 bits.
										mpz_set(large_primes[s1][0], factor1);
										mpz_set(large_primes[s1][1], factor2);
										nlp[s1] = 2;
									}
									else
										break;
								}
								else
									break;
							}
							else
								break;
						}
						else
							break;
					}
					else
						break;
#else


					if (mpz_sizeinbase(large_factors[s], 2) <= (lpb[s] * 2))
					{
						nf = mpqs_factor(large_factors[s], lpb[s], &fac);
						*num_mpqs += 1;

						for (i = 0; i < nf; i++)
						{
							if (mpz_sizeinbase(fac[i], 2) > lpb[s])
								nf = 0;
						}
					}
					else
					{
#if 0
						// try for a lucky p-1 hit on the 3LP before we go?
						// testing on an input with LPB=33 and 3LP enabled
						// saw that p-1 finds lots of factors but the residues
						// are all (99.9%) large primes.  I.e., exactly the
						// kind of inputs we want to not waste time on.
						if (getfactor_tpm1(large_factors[s], fac[0], 333))
						{
							mpz_tdiv_q(fac[1], large_factors[s], fac[0]);
							if (mpz_sizeinbase(fac[1], 2) <= lpb[s])
							{
								gmp_printf("P-1 Success! %Zd = %Zd * %Zd\n",
									large_factors[s], fac[0], fac[1]);
							}
							else if (mpz_probab_prime_p(fac[1], 1) == 0)
							{
								gmp_printf("Residue %Zd with %d bits is composite\n",
									fac[1], mpz_sizeinbase(fac[1], 2));
								gmp_printf("3LP = ");

								mpz_set(fac[2], fac[0]);
								nf = 1 + mpqs_factor(fac[2], lpb[s], &fac);

								for (i = 0; i < nf; i++)
									gmp_printf("%Zd ", fac[i]);
								printf("\n");
							}
						}
#else
						nf = 0;
#endif
					}
#endif
				}


#else

				if (mpz_sizeinbase(large_factors[s], 2) > 96)
				{
					nf = mpqs3_factor(large_factors[s], lpb[s], &fac);
					*num_mpqs3 += 1;

					for (i = 0; i < nf; i++)
					{
						if (mpz_sizeinbase(fac[i], 2) > lpb[s])
							nf = 0;
					}
				}
				else
				{
					nf = mpqs_factor(large_factors[s], lpb[s], &fac);
					*num_mpqs += 1;

					for (i = 0; i < nf; i++)
					{
						if (mpz_sizeinbase(fac[i], 2) > lpb[s])
							nf = 0;
					}
				}
#endif



			}
		}

	done:

		// _mm256_zeroupper();

		if (nf < 0)return-2;
		if (!nf)return 1;
		for (i = 0; i < nf; i++)
		{
			if (s == 0)
				mpz_set(large_primes1[nlp[s] + i], fac[i]);
			else
				mpz_set(large_primes2[nlp[s] + i], fac[i]);
		}
		nlp[s] += nf;
		done[s] = 2;

		if (done[s] < 2)return 1;

		if (s != first_side)
			break;
	}

	if ((done[0] != 2) || (done[1] != 2))return-1;

	return 0;
}

#endif