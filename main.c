
// ============================================================================
// test driver for gpu and cpu NFS 3LP relation cofactorization methods
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include "batch_factor.h"
#include "gpu_cofactorization.h"
#include <inttypes.h>
#include <string.h>
#include <immintrin.h>
#include "gmp.h"
#include "ytools.h"
#include "cmdOptions.h"
#include "arith.h"
#include <math.h>
#include "cpu_cofactorization.h"
#include "tinyecm.h"

// ============================================================================
// precision time
// ============================================================================


#if defined(WIN32) || defined(_WIN64) 
#define WIN32_LEAN_AND_MEAN

#if defined(__clang__)
#include <time.h>
#endif
#include <windows.h>
#include <process.h>
#include <winsock.h>

#else
#include <sys/time.h>	//for gettimeofday using gcc
#include <unistd.h>
#endif

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif


#ifdef _MSC_VER
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};
#endif


double _difftime(struct timeval* start, struct timeval* end);


#if defined(_MSC_VER)
int gettimeofday(struct timeval* tv, struct timezone* tz);
#endif


#if defined(_MSC_VER)

#if 0 // defined(__clang__)
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    //printf("timespec_get returned sec = %lu, nsec = %lu\n", ts.tv_sec, ts.tv_nsec);

    tv->tv_sec = ts.tv_sec;
    tv->tv_usec = ts.tv_nsec / 1000;

    return 0;
}
#else
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres /= 10;  /*convert into microseconds*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    return 0;
}
#endif
#endif

double _difftime(struct timeval* start, struct timeval* end)
{
    double secs;
    double usecs;

    if (start->tv_sec == end->tv_sec) {
        secs = 0;
        usecs = end->tv_usec - start->tv_usec;
    }
    else {
        usecs = 1000000 - start->tv_usec;
        secs = end->tv_sec - (start->tv_sec + 1);
        usecs += end->tv_usec;
        if (usecs >= 1000000) {
            usecs -= 1000000;
            secs += 1;
        }
    }

    return secs + usecs / 1000000.;
}

void mpz_to_bignum32_loc(uint32_t* bignum, mpz_t gmp_in, int words32)
{
    int i;
    mpz_t t;
    mpz_init(t);
    mpz_set(t, gmp_in);

    for (i = 0; i < words32; i++)
    {
        bignum[i] = mpz_get_ui(t) & 0xffffffff;
        mpz_tdiv_q_2exp(t, t, 32);
    }

    mpz_clear(t);
    return;
}

uint32_t process_batch(relation_batch_t *rb, int lpbr,
	int lpba, char* infile, char* outfile, int vflag, int batch_alg,
	int b1, int b2, int curves, int stop_nofactor)
{
	char buf[1024], str1[1024], str2[1024];
	uint32_t fr[32], fa[32], numr = 0, numa = 0;
	mpz_t res1, res2;
	struct timeval start;
	struct timeval stop;
	double ttime;
	uint64_t lcg_state = 0xbaddecafbaddecafull;
	int i;
	uint32_t line = 0;
	uint32_t numfull = 0;

	mpz_init(res1);
	mpz_init(res2);

	if (vflag > 0)
	{
		printf("nfs: reading input file %s...\n", infile);
	}

	FILE* fid = fopen(infile, "r");
	if (fid == NULL)
	{
		printf("could not open %s to read\n", infile);
		exit(0);
	}

	FILE* fout;

	gettimeofday(&start, NULL);

	while (~feof(fid))
	{
		int64_t a;
		uint32_t b;
		char* thistok, * nexttok;

		line++;
		char* ptr = fgets(buf, 1024, fid);
		if (ptr == NULL)
			break;

		strcpy(str1, buf);

		thistok = buf;
		nexttok = strchr(thistok, ':');
		if (nexttok == NULL)
		{
			printf("could not read relation %u, no lfactors token in file %s\n", line, infile);
			printf("line: %s\n", str1);
			continue;
		}
		*nexttok = '\0';
		nexttok++;

		ptr = strchr(thistok, ',');
		*ptr = '\0';

		mpz_set_str(res1, thistok, 10);
		mpz_set_str(res2, ptr + 1, 10);

		thistok = nexttok;
		nexttok = strchr(thistok, ':');
		if (nexttok == NULL)
		{
			printf("could not read relation %u, no a/b token in file %s\n", line, infile);
			printf("line: %s\n", str1);
			continue;
		}
		*nexttok = '\0';
		nexttok++;

		//a = strtoll(thistok, &nexttok, 10);
		//b = strtoul(nexttok + 1, &nexttok, 10);
		sscanf(thistok, "%"PRId64",%u", &a, &b);

		thistok = nexttok;
		nexttok = strchr(thistok, ':');
		if (nexttok == NULL)
		{
			printf("could not read relation %u, no rfactors token in file %s\n", line, infile);
			printf("line: %s\n", str1);
			continue;
		}
		*nexttok = '\0';
		nexttok++;


		numr = 0;
		ptr = thistok;
		while (strlen(ptr) > 0)
		{
			fr[numr++] = strtoul(ptr, NULL, 16);
			ptr = strchr(ptr, ',');
			if (ptr == NULL)
				break;
			ptr++;
		}

		thistok = nexttok;

		numa = 0;
		ptr = thistok;
		while (strlen(ptr) > 0)
		{
			fa[numa++] = strtoul(ptr, NULL, 16);
			ptr = strchr(ptr, ',');
			if (ptr == NULL)
				break;
			ptr++;
		}

		if ((mpz_sgn(res1) > 0) && (mpz_sgn(res2) > 0))
		{
			numfull++;
		}
		else
		{
			relation_batch_add(a, b, 0, fr, numr, res1, fa, numa, res2, rb);
		}
	}
	fclose(fid);

	gettimeofday(&stop, NULL);
	ttime = ytools_difftime(&start, &stop);

	if (batch_alg == 0)
	{
#ifdef HAVE_CUDA
		if (vflag >= 0)
		{
			printf("nfs: file parsing took %1.2f sec, batched %u rels. "
				"now running gpu cofactorization...\n",
				ttime, rb->num_relations);
		}

		gettimeofday(&start, NULL);

		int gpu_num = 0;
		device_ctx_t* gpu_dev_ctx = gpu_device_init(gpu_num);

		// we must create the thread context here... the cuda context
		// init method must fold in the current thread info. 
		printf("creating gpu cofactorization context\n");
		device_thread_ctx_t* gpu_cofactor_ctx =
			gpu_ctx_init(gpu_dev_ctx, rb);

		gpu_cofactor_ctx->lpba = 33;
		gpu_cofactor_ctx->lpbr = 33;
		gpu_cofactor_ctx->verbose = vflag;
		gpu_cofactor_ctx->stop_nofactor = stop_nofactor;
		do_gpu_cofactorization(gpu_cofactor_ctx, &lcg_state,
			b1, b2, 0, 0, curves, 0);

		// perhaps we can make the context persistent after we create it 
		// once in the thread?
		gpu_ctx_free(gpu_cofactor_ctx);
		gpu_dev_free(gpu_dev_ctx);

		gettimeofday(&stop, NULL);

		ttime = ytools_difftime(&start, &stop);

		if (vflag >= 0)
		{
			printf("nfs: CUDA cofactorization on %u rels from file "
				"%s took %1.4f sec producing %u relations\n",
				rb->num_relations, infile, ttime, rb->num_success);
		}
#endif
	}
	else if (batch_alg == 1)
	{
		if (vflag >= 0)
		{
			printf("nfs: file parsing took %1.2f sec, batched %u rels. "
				"now running batch solve...\n",
				ttime, rb->num_relations);
		}

		gettimeofday(&start, NULL);
		relation_batch_run(rb, rb->prime_product, &lcg_state);
		gettimeofday(&stop, NULL);

		ttime = ytools_difftime(&start, &stop);

		if (vflag >= 0)
		{
			printf("nfs: relation_batch_run on %u rels from file "
				"%s took %1.4f sec producing %u relations\n",
				rb->num_relations, infile, ttime, rb->num_success);
		}
	}
	else if ((batch_alg == 2) || (batch_alg == 3))
	{
		if (vflag >= 0)
		{
			printf("nfs: file parsing took %1.2f sec, batched %u rels. "
				"now running cofactorization loop...\n",
				ttime, rb->num_relations);
		}

		gettimeofday(&start, NULL);
		
		mpz_t large_factors[2];
		mpz_t large_primes1[3];
		mpz_t large_primes2[3];

		mpz_init(large_factors[0]);
		mpz_init(large_factors[1]);
		mpz_init(large_primes1[0]);
		mpz_init(large_primes1[1]);
		mpz_init(large_primes2[0]);
		mpz_init(large_primes2[1]);

		int total_factors = 0;
		int total_ecm64 = 0;
		int total_ecm128 = 0;
		int total_mpqs = 0;
		int total_mpqs3 = 0;
		for (i = 0; i < rb->num_relations; i++)
		{
			uint16_t lpb[2] = { 33, 33 };
			uint32_t nlp[2];
			int j;

			// translate the relation_batch format to 
			// mpz_t's we can send to the cofactorization routine.
			cofactor_t* c = &rb->relations[i];

			uint32_t* f = rb->factors + c->factor_list_word;
			uint32_t* lp1 = f + c->num_factors_r + c->num_factors_a;
			uint32_t* lp2 = lp1 + c->lp_r_num_words;

			if (c->lp_r_num_words)
			{
				mpz_set_ui(large_factors[0], lp1[c->lp_r_num_words - 1]);
				for (j = c->lp_r_num_words - 2; j >= 0; j--)
				{
					mpz_mul_2exp(large_factors[0], large_factors[0], 32);
					mpz_add_ui(large_factors[0], large_factors[0], lp1[j]);
				}
				mpz_neg(large_factors[0], large_factors[0]);
			}
			else if (c->lp_r[0] > 0)
			{
				mpz_set_ui(large_factors[0], c->lp_r[0]);
			}
			else
			{
				mpz_set_ui(large_factors[0], 1);
			}

			if (c->lp_a_num_words) {

				mpz_set_ui(large_factors[1], lp2[c->lp_a_num_words - 1]);
				for (j = c->lp_a_num_words - 2; j >= 0; j--)
				{
					mpz_mul_2exp(large_factors[1], large_factors[1], 32);
					mpz_add_ui(large_factors[1], large_factors[1], lp2[j]);
				}
				mpz_neg(large_factors[1], large_factors[1]);
			}
			else if (c->lp_a[0] > 0)
			{
				mpz_set_ui(large_factors[1], c->lp_a[0]);
			}
			else
			{
				mpz_set_ui(large_factors[1], 1);
			}

			if (i % 65536 == 0)
			{
				gmp_printf("now processing relation %d: %Zd, %Zd\n",
					i, large_factors[0], large_factors[1]);
			}

			mpz_t* lfs = large_factors;
			mpz_t* lps1 = large_primes1;
			mpz_t* lps2 = large_primes2;

			int num_ecm64 = 0;
			int num_ecm128 = 0;
			int num_mpqs = 0;
			int num_mpqs3 = 0;
			int status;

			// test of cpu p-1
			//mpz_set_ui(large_primes1[0], 0);
			//
			//if (mpz_sgn(large_factors[1]) < 0)
			//{
			//	mpz_neg(large_factors[1], large_factors[1]);
			//	getfactor_tpm1(large_factors[1], large_primes1[0], 333);
			//
			//	if (mpz_cmp_ui(large_primes1[0], 1) > 0)
			//	{
			//		//gmp_printf("found factor %Zd of %Zd with P-1\n",
			//		//	large_primes1[0], large_factors[1]);
			//		rb->num_success++;
			//	}
			//}
			//
			//continue;
			
			// its faster to start on the 2LP side, have fewer
			// more-difficult 64b+ inputs to consider.
			if (batch_alg == 2)
			{
				// ecm + mpqs
				status = cofactorisation(0, lps1, lps2, lfs, lpb, nlp, 0,
					&num_mpqs, &num_mpqs3, &num_ecm64, &num_ecm128);
			}
			else
			{
				// mpqs only
				status = cofactorisation(0, lps1, lps2, lfs, lpb, nlp, 1,
					&num_mpqs, &num_mpqs3, &num_ecm64, &num_ecm128);
			}

			total_ecm64 += num_ecm64;
			total_ecm128 += num_ecm128;
			total_mpqs += num_mpqs;
			total_mpqs3 += num_mpqs3;

			total_factors += nlp[0];
			total_factors += nlp[1];

			if (status == 0)
			{
				rb->num_success++;
			}
		}

		gettimeofday(&stop, NULL);

		ttime = ytools_difftime(&start, &stop);

		if (vflag >= 0)
		{
			printf("nfs: cofactorization on %u rels from file "
				"%s took %1.4f sec producing %u relations\n",
				rb->num_relations, infile, ttime, rb->num_success);

			printf("subroutine call stats:\n");
			printf("\tnum_ecm64 = %d\n", total_ecm64);
			printf("\tnum_ecm128 = %d\n", total_ecm128);
			printf("\tnum_mpqs = %d\n", total_mpqs);
			printf("\tnum_mpqs3 = %d\n", total_mpqs3);
		}

	}


	if ((vflag > 0) && (batch_alg == 1))
	{
		printf("ECM stats R:\n");
		for (i = 0; i < 4; i++)
		{
			printf("%u;  ", rb->num_uecm[i]);
		}
		printf("%u;  ", rb->num_tecm);
		printf("%u;  ", rb->num_tecm2);
		printf("%u;  ", rb->num_qs);
		printf("\nECM stats A:\n");
		for (i = 0; i < 4; i++)
		{
			printf("%u;  ", rb->num_uecm_a[i]);
		}
		printf("%u;  ", rb->num_tecm_a);
		printf("%u;  ", rb->num_tecm2_a);
		printf("%u;  ", rb->num_qs_a);

		printf("\nAbort stats R:\n");
		for (i = 0; i < 8; i++)
		{
			printf("%u;  ", rb->num_abort[i]);
		}
		printf("\nAbort stats A:\n");
		for (i = 0; i < 8; i++)
		{
			printf("%u;  ", rb->num_abort_a[i]);
		}
		printf("\n");
	}

	mpz_clear(res1);
	mpz_clear(res2);

	return rb->num_success;
}

int main(int argc, char **argv) {
    char fname[80];
	int batch_alg = 0;
	int lpbr = 33;
	int lpba = 33;
	relation_batch_t rb;
	options_t* options = initOpt();

	processOpts(argc, argv, options);
	batch_alg = options->batch_method;

    strcpy(fname, options->file);

	if (0)
	{
		mpz_t n, f;
		mpz_init(n);
		mpz_init(f);
	
		mpz_set_str(n, "41716014795600569829721264369", 10);
		getfactor_tpm1(n, f, 500);
		gmp_printf("%Zd\n", f);

		mpz_set_str(n, "12525831385794046220132818133", 10);
		getfactor_tpm1(n, f, 500);
		gmp_printf("%Zd\n", f);

		mpz_clear(n);
		mpz_clear(f);
	}

	if (batch_alg == 1)
	{
		char fname[80];
		sprintf(fname, "bgcd_lpb%d", MAX(lpbr, lpba));
		FILE* fid = fopen(fname, "rb");
		int compute_pproduct = 1;

		if (fid != NULL)
		{
			compute_pproduct = 0;
		}

		// this initializes the prime product, regardless of whether it is computed or not.
		relation_batch_init(stdout, &rb, 1000000, 1ULL << ((MAX(lpbr, lpba) - 1)),
			1ull << lpbr, 1ull << lpba, NULL, compute_pproduct);

		if (fid != NULL)
		{
			mpz_inp_raw(rb.prime_product, fid);

			printf("loaded prime product from file %s: product has %"PRIu64" bits\n",
				fname, (uint64_t)mpz_sizeinbase(rb.prime_product, 2));

			fclose(fid);
		}

		if (compute_pproduct == 1)
		{
			printf("exporting prime product to file %s; approx file size = %u MB\n", fname,
				(uint32_t)(mpz_sizeinbase(rb.prime_product, 2) / 8 / (1 << 20)));

			// Make the file for future use.
			fid = fopen(fname, "wb");
			mpz_out_raw(fid, rb.prime_product);
			fclose(fid);
		}
	}
	else if (batch_alg == 0)
	{
		relation_batch_init(stdout, &rb, 1000000, 1ULL << (MAX(lpbr, lpba) - 1),
			1ull << lpbr, 1ull << lpba, NULL, 0);
	}
	else if (batch_alg == 2)
	{
		// normal cofactorization, ecm + mpqs
		relation_batch_init(stdout, &rb, 1000000, 1ULL << (MAX(lpbr, lpba) - 1),
			1ull << lpbr, 1ull << lpba, NULL, 0);
	}
	else if (batch_alg == 3)
	{
		// normal cofactorization, mpqs only
		relation_batch_init(stdout, &rb, 1000000, 1ULL << (MAX(lpbr, lpba) - 1),
			1ull << lpbr, 1ull << lpba, NULL, 0);
	}
	else
	{
		printf("-m (--method) must be 0 for GPU, 1 for batch GCD, 2 for CPU (ecm+mpqs)"
			", or 3 for CPU (mpqs only)\n");
		exit(0);
	}

    process_batch(&rb, lpbr, lpba, options->file, "", 1, batch_alg,
		options->b1_3lp, options->b2_3lp, options->curves_3lp, options->stop_nofactor);

    return 0;
}
