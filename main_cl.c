
// ============================================================================
// test driver for gpu and cpu NFS 3LP relation cofactorization methods
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include "batch_factor.h"
#include "gpu_cofactorization_cl.h"
#include <inttypes.h>
#include <string.h>
#include <immintrin.h>
#include "gmp.h"
#include "ytools.h"
#include "cmdOptions.h"
#include "arith.h"
#include <math.h>
#include "tinyecm.h"
#include "microecm.h"

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

#if defined(__MINGW32__)
#include <sys/time.h>
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

void discard_unsuccessful(relation_batch_t* rb)
{
	int i;
	int j;

	for (i = 0, j = 0; i < rb->num_relations; i++)
	{
		if (rb->relations[i].success)
		{
			// int k;
			// if (i < 10)
			// {
			// 	printf("%d: %"PRId64", %u:\n", i, rb->relations[i].a, rb->relations[i].b);
			// 	for (k = 0; k < MAX_LARGE_PRIMES; k++)
			// 	{
			// 		if (rb->relations[i].lp_r[k] > 1)
			// 			printf("%"PRIu64",", rb->relations[i].lp_r[k]);
			// 	}
			// 	printf(":");
			// 	for (k = 0; k < MAX_LARGE_PRIMES; k++)
			// 	{
			// 		if (rb->relations[i].lp_a[k] > 1)
			// 			printf("%"PRIu64",", rb->relations[i].lp_a[k]);
			// 	}
			// 	printf("\n");
			// }
			// copy to next position in output batch
			memcpy(&rb->relations[j], &rb->relations[i], sizeof(cofactor_t));
			j++;
		}
	}
	rb->num_relations = j;
	return;
}

int cmp_relation(const void* x, const void* y)
{
	cofactor_t* xx = (cofactor_t*)x;
	cofactor_t* yy = (cofactor_t*)y;

	if (xx->a > yy->a)
		return 1;
	else if (xx->a == yy->a)
		return 0;
	else
		return -1;
}

int qcomp_uint64_dsc(const void* x, const void* y)
{
	uint64_t* xx = (uint64_t*)x;
	uint64_t* yy = (uint64_t*)y;

	if (*xx < *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

void print_relations(relation_batch_t* rb, char* filename)
{
	int i;
	FILE* fid = fopen(filename, "w");

	if (fid == NULL)
		return;

	for (i = 0; i < rb->num_relations; i++)
	{
		int j;

		fprintf(fid, "%"PRId64",%u:", rb->relations[i].a, rb->relations[i].b);
		qsort(rb->relations[i].lp_r, MAX_LARGE_PRIMES, sizeof(uint64_t), &qcomp_uint64_dsc);
		for (j = 0; j < MAX_LARGE_PRIMES; j++)
		{
			if (rb->relations[i].lp_r[j] > 1)
				fprintf(fid, "%"PRIu64",", rb->relations[i].lp_r[j]);
		}
		fprintf(fid, ":");
		qsort(rb->relations[i].lp_a, MAX_LARGE_PRIMES, sizeof(uint64_t), &qcomp_uint64_dsc);
		for (j = 0; j < MAX_LARGE_PRIMES; j++)
		{
			if (rb->relations[i].lp_a[j] > 1)
				fprintf(fid, "%"PRIu64",", rb->relations[i].lp_a[j]);
		}
		fprintf(fid, "\n");
	}

	fclose(fid);
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
#ifdef HAVE_CUDA_BATCH_FACTOR
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

		gpu_cofactor_ctx->lpba = rb->lpba;
		gpu_cofactor_ctx->lpbr = rb->lpbr;
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
			printf("nfs: OpenCL cofactorization on %u rels from file "
				"%s took %1.4f sec producing %u relations\n",
				rb->num_relations, infile, ttime, rb->num_success);
		}

		// write the processed relations.  Sorted, so we can compare lists
		// produced by different settings or strategies or tools.
		printf("writing sorted output... ");

		discard_unsuccessful(rb);
		qsort(rb->relations, rb->num_relations, sizeof(cofactor_t), &cmp_relation);	// sort by A

		char outfile[80];
		sprintf(outfile, "%s.cl.out", infile);
		print_relations(rb, outfile);

		printf("done\n");

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
	int lpbr = 31;
	int lpba = 31;
	relation_batch_t rb;
	options_t* options = initOpt();

	processOpts(argc, argv, options);
	batch_alg = options->batch_method;
	lpbr = rb.lpbr = options->lpbr;
	lpba = rb.lpba = options->lpba;

    strcpy(fname, options->file);

	if (batch_alg == 1)
	{
		char fname[80];

		// choose how big to make the GCD prime product.
		// anecdotally, as lpb increases, it becomes more
		// rare for all factors to be simultaneously large,
		// so we can reduce the product by an extra bit without
		// loosing too many of these splits.  The smaller
		// GCD is a lot faster.
		int file_bits = MAX(lpbr, lpba);
		
		if (file_bits > 32)
			file_bits -= 2;
		else
			file_bits -= 1;

		sprintf(fname, "bgcd_lpb%d", file_bits);
		FILE* fid = fopen(fname, "rb");
		int compute_pproduct = 1;

		if (fid != NULL)
		{
			compute_pproduct = 0;
		}

		// this initializes the prime product, regardless of whether it is computed or not.
		relation_batch_init(stdout, &rb, 10000000, 1ULL << file_bits,
			1ull << lpbr, 1ull << lpba, NULL, compute_pproduct);

		if (fid != NULL)
		{
			mpz_inp_raw(rb.prime_product, fid);

			printf("loaded prime product from file %s: product has %"PRIu64" bits\n",
				fname, (uint64_t)mpz_sizeinbase(rb.prime_product, 2));
			printf("memory use is %u MB\n", 
				(uint64_t)mpz_sizeinbase(rb.prime_product, 2) / 8 / (1 << 20));

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
	else
	{
		printf("-m (--method) must be 0 for GPU or 1 for batch GCD\n");
		exit(0);
	}

    process_batch(&rb, lpbr, lpba, options->file, "", 1, batch_alg,
		options->b1_3lp, options->b2_3lp, options->curves_3lp, options->stop_nofactor);

    return 0;
}
