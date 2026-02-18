
// ============================================================================
// main.c - Pure C application code
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

uint32_t loc_sieve(uint32_t limit, uint32_t* primes)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.
	uint8_t* flags;
	uint32_t prime;
	uint32_t i, j;
	int it;

	//allocate flags
	flags = (uint8_t*)xmalloc(limit / 2 * sizeof(uint8_t));
	memset(flags, 1, limit / 2);

	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	primes[0] = 2;
	it = 1;

	//sieve using primes less than the sqrt of the desired limit
	//flags are created only for odd numbers (mod2)
	for (i = 1; i < (uint32_t)(sqrt(limit) / 2 + 1); i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint32_t)(2 * i + 1);
			for (j = i + prime; j < limit / 2; j += prime)
				flags[j] = 0;

			primes[it] = prime;
			it++;
		}
	}

	//now find the rest of the prime flags and compute the sieving primes
	for (; i < limit / 2; i++)
	{
		if (flags[i] == 1)
		{
			primes[it] = (uint32_t)(2 * i + 1);
			it++;
		}
	}

	free(flags);
	return it;
}


#define ADD 6.0
#define DUP 4.0
#define NV 10  

#define NUMP_SMALLP 1438
static const int primes[NUMP_SMALLP] = {
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087,
1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153,
1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229,
1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297,
1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381,
1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453,
1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523,
1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597,
1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663,
1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741,
1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823,
1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901,
1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993,
1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063,
2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131,
2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221,
2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293,
2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371,
2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437,
2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539,
2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621,
2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689,
2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749,
2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833,
2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909,
2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001,
3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259,
3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433,
3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517,
3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581,
3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659,
3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733,
3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823,
3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911,
3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001,
4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073,
4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153,
4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241,
4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327,
4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421,
4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507,
4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591,
4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663,
4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759,
4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861,
4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943,
4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009,
5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099,
5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189,
5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281,
5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393,
5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449,
5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527,
5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641,
5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701,
5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801,
5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861,
5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953,
5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067,
6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143,
6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229,
6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311,
6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373,
6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481,
6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577,
6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679,
6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763,
6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841,
6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947,
6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001,
7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109,
7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211,
7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307,
7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417,
7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507,
7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573,
7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649,
7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727,
7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841,
7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927,
7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039,
8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117,
8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221,
8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293,
8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389,
8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513,
8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599,
8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681,
8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747,
8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, 8837,
8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933,
8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013,
9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127,
9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203,
9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293,
9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391,
9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461,
9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539,
9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643,
9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739,
9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817,
9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901,
9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973, 10007, 10009,
10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111,
10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211,
10223, 10243, 10247, 10253, 10259, 10267, 10271, 10273, 10289, 10301, 10303,
10313, 10321, 10331, 10333, 10337, 10343, 10357, 10369, 10391, 10399, 10427,
10429, 10433, 10453, 10457, 10459, 10463, 10477, 10487, 10499, 10501, 10513,
10529, 10531, 10559, 10567, 10589, 10597, 10601, 10607, 10613, 10627, 10631,
10639, 10651, 10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729,
10733, 10739, 10753, 10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853,
10859, 10861, 10867, 10883, 10889, 10891, 10903, 10909, 10937, 10939, 10949,
10957, 10973, 10979, 10987, 10993, 11003, 11027, 11047, 11057, 11059, 11069,
11071, 11083, 11087, 11093, 11113, 11117, 11119, 11131, 11149, 11159, 11161,
11171, 11173, 11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261, 11273,
11279, 11287, 11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383,
11393, 11399, 11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489,
11491, 11497, 11503, 11519, 11527, 11549, 11551, 11579, 11587, 11593, 11597,
11617, 11621, 11633, 11657, 11677, 11681, 11689, 11699, 11701, 11717, 11719,
11731, 11743, 11777, 11779, 11783, 11789, 11801, 11807, 11813, 11821, 11827,
11831, 11833, 11839, 11863, 11867, 11887, 11897, 11903, 11909, 11923, 11927,
11933, 11939, 11941, 11953, 11959, 11969, 11971, 11981, 11987};

double getEcost(uint64_t d, uint64_t e)
{
	int doub = 0, add = 0;

	while (d > 0)
	{
		if ((e / 2) < d)
		{
			d = e - d;
		}
		else if ((d < (e / 4)) && ((e & 1) == 0))
		{
			e = e / 2;
			doub++;
			add++;
		}
		else
		{
			e = e - d;
			add++;
		}

	}
	return (doub + add) * 2 + add * 4 + doub * 3;
}

static double lucas_cost(uint64_t n, double v)
{
	uint64_t d, e, r;
	double c; /* cost */

	if (n == 0)
	{
		printf("lucas input cannot be 0\n");
		exit(1);
	}

	d = n;
	r = (uint64_t)((double)d * v + 0.5);
	if (r >= n)
		return (ADD * (double)n);
	d = n - r;
	e = 2 * r - n;
	c = DUP + ADD; /* initial duplicate and final addition */
	while (d != e)
	{
		if (d < e)
		{
			r = d;
			d = e;
			e = r;
		}
		if (d - e <= e / 4 && ((d + e) % 3) == 0)
		{ /* condition 1 */
			d = (2 * d - e) / 3;
			e = (e - d) / 2;
			c += 3.0 * ADD; /* 3 additions */
		}
		else if (d - e <= e / 4 && (d - e) % 6 == 0)
		{ /* condition 2 */
			d = (d - e) / 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		else if ((d + 3) / 4 <= e)
		{ /* condition 3 */
			d -= e;
			c += ADD; /* one addition */
		}
		else if ((d + e) % 2 == 0)
		{ /* condition 4 */
			d = (d - e) / 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		/* now d+e is odd */
		else if (d % 2 == 0)
		{ /* condition 5 */
			d /= 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		/* now d is odd and e is even */
		else if (d % 3 == 0)
		{ /* condition 6 */
			d = d / 3 - e;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else if ((d + e) % 3 == 0)
		{ /* condition 7 */
			d = (d - 2 * e) / 3;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else if ((d - e) % 3 == 0)
		{ /* condition 8 */
			d = (d - e) / 3;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else /* necessarily e is even: catches all cases */
		{ /* condition 9 */
			e /= 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
	}

	if (d != 1)
	{
		c = 9999999.;
	}

	return c;
}

void lucas_opt(uint32_t B1, int num)
{
	uint64_t d, e, r, q, c;
	double cmin = 99999999., cost;
	int best_index = 0;
	int i, j;
	gmp_randstate_t gmprand;
	double* val;
	double trad_cost;
	uint64_t word;
	double* v;
	double* pcost;
	uint64_t f[128];
	int nump;

	gmp_randinit_default(gmprand);

	val = (double*)malloc((num + NV) * sizeof(double));
	val[0] = 0.61803398874989485;
	val[1] = 0.72360679774997897;
	val[2] = 0.58017872829546410;
	val[3] = 0.63283980608870629;
	val[4] = 0.61242994950949500;
	val[5] = 0.62018198080741576;
	val[6] = 0.61721461653440386;
	val[7] = 0.61834711965622806;
	val[8] = 0.61791440652881789;
	val[9] = 0.61807966846989581;

	for (i = NV; i < (num + NV); i++)
	{
		uint64_t ur = gmp_urandomb_ui(gmprand, 52);
		double r = 0.51 + ((double)ur * 2.2204460492503130808472633361816e-16) * 0.25;
		val[i] = r;
	}

	c = 1;
	e = 1;
	j = 0;
	i = 2;
	q = primes[i];
	trad_cost = 0.;

	while (q < 1000)
	{
		i++;
		q = primes[i];
	}
	nump = i;

	v = (double*)malloc(i * sizeof(double));
	pcost = (double*)malloc(i * sizeof(double));

	i = 2;
	q = primes[i];
	pcost[0] = 5.0;
	pcost[1] = 11.0;

	while (q < 1000)
	{
		printf("now optimizing prime %lu\n", q);
		for (d = 0, cmin = ADD * (double)q; d < (num + NV); d++)
		{
			cost = lucas_cost(q, val[d]);
			if (cost < cmin)
			{
				cmin = cost;
				best_index = d;
				printf("best cost is now %1.6f at index %d\n", cmin, d);
			}
		}

		printf("minimum prac cost of prime %lu is %1.6f at index %d, val %1.18f\n",
			q, cmin, best_index, val[best_index]);

		pcost[i] = cmin;
		v[i] = val[best_index];

		i++;
		q = primes[i];

	}

	printf("now optimizing composites\n");
	c = 6;
	while (c < 1000000)
	{
		int skip = 0;
		double fcost;

		j = 2;
		q = primes[j];
		while (q < c)
		{
			q = primes[j++];
			if (c == q)
			{
				skip = 1;
				break;
			}
		}

		if (skip)
		{
			c++;
			continue;
		}

		e = c;
		while ((e & 1) == 0)
			e /= 2;

		//printf("now optimizing composite %lu\n", e);
		for (d = 0, cmin = ADD * (double)e; d < (num + NV); d++)
		{
			cost = lucas_cost(e, val[d]);
			if (cost < cmin)
			{
				cmin = cost;
				best_index = d;
				//printf("best cost is now %1.6f at index %d\n", cmin, d);
			}
		}

		fcost = 0.0;
		d = e;
		j = 0;
		for (i = 0; i < nump; i++)
		{
			if (d % primes[i] == 0)
			{
				fcost += pcost[i];
				d /= primes[i];
				f[j++] = primes[i];
			}

			if (d == 1)
				break;
		}

		if (fcost > cmin)
		{
			printf("minimum prac cost of %lu is %1.1f at index %d, "
				"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
			for (i = 0; i < j; i++)
				printf("%lu,", f[i]);
			printf("\b] is %1.1f\n", fcost);
			fflush(stdout);
		}
		c++;

	}

	exit(0);

	while (q < B1)
	{
		//printf("now accumulating prime %lu, word is %lu\n", q, c);
		if ((e * q) < B1)
		{
			// add another of this prime to the word
			// as long as it still fits in the word.
			if ((0xFFFFFFFFFFFFFFFFULL / c) > q)
			{
				c *= q;
				e *= q;

				//printf("now optimizing traditional word %lu\n", q);
				for (d = 0, cmin = ADD * (double)q; d < (num + NV); d++)
				{
					cost = lucas_cost(q, val[d]);
					if (cost < cmin)
					{
						cmin = cost;
						best_index = d;
					}
				}

				trad_cost += cmin;
			}
			else
			{

				// word is full, proceed to optimize
				printf("now optimizing word %d = %lu\n", j, c);

				for (d = 0, cmin = ADD * (double)c; d < (num + NV); d++)
				{
					cost = lucas_cost(c, val[d]);
					if (cost < cmin)
					{
						cmin = cost;
						best_index = d;
						//printf("best cost is now %1.6f at index %d\n", cmin, d);
					}
				}

				printf("minimum prac cost is %1.6f at index %d, val %1.18f\n",
					cmin, best_index, val[best_index]);

				for (d = 0, cmin = ADD * (double)c; d < (num + NV); d++)
				{
					uint64_t dd = (uint64_t)((double)c * val[d]);
					uint64_t ee = c;

					if (spGCD(dd, ee) != 1)
					{
						continue;
					}

					cost = getEcost(dd, ee);
					if (cost < cmin)
					{
						cmin = cost;
						best_index = d;
					}

				}

				printf("minimum euclid cost is %1.6f at index %d, val %1.18f\n",
					cmin, best_index, val[best_index]);
				printf("traditional cost would have been: %1.6f\n", trad_cost);

				// next word
				trad_cost = 0.;
				c = 1;
				e = 1;
				j++;
			}
		}
		else
		{
			// next prime
			i++;
			e = 1;
			q = primes[i];
		}

	}

	printf("final word %d is %lu\n", j, c);
	for (d = 0, cmin = ADD * (double)c; d < (num + NV); d++)
	{
		cost = lucas_cost(c, val[d]);
		if (cost < cmin)
		{
			cmin = cost;
			best_index = d;
		}
	}

	printf("minimum prac cost is %1.6f at index %d, val %1.18f\n",
		cmin, best_index, val[best_index]);

	for (d = 0, cmin = ADD * (double)c; d < (num + NV); d++)
	{
		uint64_t dd = (uint64_t)((double)c * val[d]);
		uint64_t ee = c;

		if (spGCD(dd, ee) != 1)
		{
			continue;
		}

		cost = getEcost(dd, ee);
		if (cost < cmin)
		{
			cmin = cost;
			best_index = d;
		}

	}

	printf("minimum euclid cost is %1.6f at index %d, val %1.18f\n",
		cmin, best_index, val[best_index]);
	printf("traditional cost would have been: %1.6f\n", trad_cost);

	free(val);
	return;
}

void lucas_opt2(uint32_t B1, int num)
{
	uint64_t d, e, r, q, c;
	double cmin = 99999999., cost;
	int best_index = 0;
	int i, j, k, l;
	gmp_randstate_t gmprand;
	double* val;
	double trad_cost;
	uint64_t word;
	double* v;
	double* pcost;
	uint64_t f[128];
	int nump;

	gmp_randinit_default(gmprand);

	val = (double*)malloc((num + NV) * sizeof(double));
	val[0] = 0.61803398874989485;
	val[1] = 0.72360679774997897;
	val[2] = 0.58017872829546410;
	val[3] = 0.63283980608870629;
	val[4] = 0.61242994950949500;
	val[5] = 0.62018198080741576;
	val[6] = 0.61721461653440386;
	val[7] = 0.61834711965622806;
	val[8] = 0.61791440652881789;
	val[9] = 0.61807966846989581;

	// get other random values to check centered around val[0].
	double range = 0.2;
	for (i = NV; i < (num + NV); i++)
	{
		uint64_t ur = gmp_urandomb_ui(gmprand, 52);
		double r = (val[0] - range/2.0) + 
			((double)ur * 2.2204460492503130808472633361816e-16) * range;
		val[i] = r;
	}

	c = 1;
	e = 1;
	j = 0;
	i = 2;
	q = primes[i];
	trad_cost = 0.;

	while (q < B1)
	{
		i++;
		q = primes[i];
	}
	nump = i;

	v = (double*)malloc(i * sizeof(double));
	pcost = (double*)malloc(i * sizeof(double));

	double total_cost = 0.0;
	i = 1;
	q = primes[i];
	pcost[0] = 5.0;
	pcost[1] = 11.0;
	total_cost = 16.0;

	FILE* fid;
	fid = fopen("lucas_cost_comb.txt", "w");

	while (q < B1)
	{
		//printf("now optimizing prime %lu\n", q);
		for (d = 0, cmin = ADD * (double)q; d < (num + NV); d++)
		{
			cost = lucas_cost(q, val[d]);
			if (cost < cmin)
			{
				cmin = cost;
				best_index = d;
				//printf("best cost is now %1.6f at index %d\n", cmin, d);
			}
		}

		printf("minimum prac cost of prime %lu is %1.6f at index %d, val %1.18f\n",
			q, cmin, best_index, val[best_index]);
		fprintf(fid,"minimum prac cost of prime %lu is %1.6f at index %d, val %1.18f\n",
			q, cmin, best_index, val[best_index]);

		total_cost += cmin;
		pcost[i] = cmin;
		v[i] = val[best_index];

		i++;
		q = primes[i];
	}

	int max_idx = i - 1;

	printf("total B1 cost (up to prime index %u) is %1.0f\n", max_idx, total_cost);
	fprintf(fid,"total B1 cost (up to prime index %u) is %1.0f\n", max_idx, total_cost);

	if (1) {
		printf("now optimizing pairs of primes\n");
		fprintf(fid,"now optimizing pairs of primes\n");
		double total_savings = 0.0;
		i = 0;
		{
			j = 0;
			{
				for (k = j + 1; k < max_idx; k++)
				{
					for (l = k + 1; l < max_idx; l++)
					{
						double fcost;
						e = primes[k] * primes[l];

						//int maxd = (num + NV);
						int maxd = NV;
						for (d = 0, cmin = ADD * (double)e; d < maxd; d++)
						{
							cost = lucas_cost(e, val[d]);
							if (cost < cmin)
							{
								cmin = cost;
								best_index = d;
								//printf("best cost is now %1.6f at index %d\n", cmin, d);
							}
						}

						fcost = pcost[k] + pcost[l];

						if (cmin < fcost)
						{
							double savings = fcost - cmin;

							if (savings > 3.0)
							{
								
								fprintf(fid,"minimum prac cost of %lu is %1.0f at index %d, "
									"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
								//printf("%lu,", f[0]);
								//printf("%lu,", f[1]);
								fprintf(fid,"%lu,", primes[k]);
								fprintf(fid,"%lu,", primes[l]);
								fprintf(fid,"\b] is %1.0f + %1.0f = %1.0f, savings = %1.0f\n",
									pcost[k], pcost[l], fcost, fcost - cmin);
								//fflush(stdout);
							}

							total_savings += (fcost - cmin);
						}
					}
				}
			}
		}

		printf("total savings is %1.0f\n", total_savings);
		fprintf(fid,"total savings is %1.0f\n", total_savings);
		
	}

	if (0) {
		printf("now optimizing triples of primes\n");
		double total_savings = 0.0;
		i = 0;
		{
			for (j = i + 1; j < max_idx; j++)
			{
				for (k = j + 1; k < max_idx; k++)
				{
					for (l = k + 1; l < max_idx; l++)
					{
						double fcost;
						e = primes[j] * primes[k] * primes[l];

						for (d = 0, cmin = ADD * (double)e; d < (num + NV); d++)
						{
							cost = lucas_cost(e, val[d]);
							if (cost < cmin)
							{
								cmin = cost;
								best_index = d;
								//printf("best cost is now %1.6f at index %d\n", cmin, d);
							}
						}

						fcost = pcost[j] + pcost[k] + pcost[l];

						if (cmin < fcost)
						{
							//FILE* fid;
							//fid = fopen("lucas_cost_comb.txt", "a");
							printf("minimum prac cost of %lu is %1.0f at index %d, "
								"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
							//printf("%lu,", f[0]);
							printf("%lu,", primes[j]);
							printf("%lu,", primes[k]);
							printf("%lu,", primes[l]);
							printf("\b] is %1.0f, savings = %1.0f\n", fcost, fcost - cmin);
							fflush(stdout);

							total_savings += (fcost - cmin);

							//fprintf(fid, "minimum prac cost of %lu is %1.1f at index %d, "
							//	"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
							//fprintf(fid, "%lu,", f[0]);
							//fprintf(fid, "%lu,", f[1]);
							//fprintf(fid, "%lu,", f[2]);
							//fprintf(fid, "%lu,", f[3]);
							//fprintf(fid, "\b] is %1.1f\n", fcost);
							//fclose(fid);
						}
					}
				}
			}
		}

		printf("total savings is %1.0f\n", total_savings);
	}


	if (0) {
		printf("now optimizing composites composed of combinations of 4 primes\n");
		for (i = 3; i < 19; i++)
		{
			for (j = i + 1; j < 19; j++)
			{
				for (k = j + 1; k < 19; k++)
				{
					for (l = k + 1; l < 19; l++)
					{
						double fcost;
						e = primes[i] * primes[j] * primes[k] * primes[l];

						printf("now optimizing composite %lu = %u * %u * %u * %u\n",
							e, primes[i], primes[j], primes[k], primes[l]);
						//printf("now optimizing composite %lu = %u * %u * %u\n",
					//		e, primes[i], primes[j], primes[k]); 

						for (d = 0, cmin = ADD * (double)e; d < (num + NV); d++)
						{
							cost = lucas_cost(e, val[d]);
							if (cost < cmin)
							{
								cmin = cost;
								best_index = d;
								//printf("best cost is now %1.6f at index %d\n", cmin, d);
							}
						}

						fcost = pcost[i] + pcost[j] + pcost[k] + pcost[l];
						f[0] = primes[i];
						f[1] = primes[j];
						f[2] = primes[k];
						f[3] = primes[l];

						if (fcost > cmin)
						{
							FILE* fid;
							fid = fopen("lucas_cost_comb.txt", "a");
							printf("minimum prac cost of %lu is %1.1f at index %d, "
								"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
							printf("%lu,", f[0]);
							printf("%lu,", f[1]);
							printf("%lu,", f[2]);
							printf("%lu,", f[3]);
							printf("\b] is %1.1f\n", fcost);
							fflush(stdout);

							fprintf(fid, "minimum prac cost of %lu is %1.1f at index %d, "
								"val %1.18f, factor cost of [", e, cmin, best_index, val[best_index]);
							fprintf(fid, "%lu,", f[0]);
							fprintf(fid, "%lu,", f[1]);
							fprintf(fid, "%lu,", f[2]);
							fprintf(fid, "%lu,", f[3]);
							fprintf(fid, "\b] is %1.1f\n", fcost);
							fclose(fid);
						}
					}
				}
			}
		}
	}

	fclose(fid);
	exit(0);
	free(val);
	return;
}

void lucas_opt3(uint32_t B1, int numExtraV, int numSeeds)
{
	// to make this an implementable thing with large B1, we run
	// the following optimization:
	// generate 'num' extra V's.
	// for each seed out of numSeeds, shuffle the indices of the primes
	// making up B1.
	// group the shuffled primes in groups of 2, 3, 4, ...
	// get the cost of each group relative to its constituent primes
	// for each V.
	// find the best V for each group and the best seed overall.
	// then for each B1, we'd have to store the seed and list of V's.
	// generate the primes, shuffle them, and send prime list and V list
	// to gpu global memory.
	// after playing with this a bit it's apparent that optimizing will
	// only save ~1% of the cost compared to just using the original
	// 10 V's on each constituent prime.  more V's is more beneficial than
	// pairing, but again only to within 1% of doing nothing.
	// look at 4-NAF.

	uint64_t d, e, r, q, c;
	double cmin = 99999999., cost;
	int best_index = 0;
	int i, j, k, l;
	gmp_randstate_t gmprand;
	double* val;
	double trad_cost;
	uint64_t word;
	double* v;
	double* pcost;
	int nump;
	mpz_t rndn, maxn;
	mpz_init(rndn);
	mpz_init(maxn);
	int *pshuf;
	int grpsz = 2;
	double total_cost = 0.0;
	int num_groups = 0;
	int *grp_idx = (int*)malloc(B1 * sizeof(int));
	int* grp_sz = (int*)malloc(B1 * sizeof(int));
	double* grp_cost = (double *)malloc(B1 * sizeof(double));

	uint32_t* plist = (uint32_t *)malloc(1000000 * sizeof(uint32_t));
	loc_sieve(1000100, plist);

	gmp_randinit_default(gmprand);

	val = (double*)malloc((numExtraV + NV) * sizeof(double));
	val[0] = 0.61803398874989485;
	val[1] = 0.72360679774997897;
	val[2] = 0.58017872829546410;
	val[3] = 0.63283980608870629;
	val[4] = 0.61242994950949500;
	val[5] = 0.62018198080741576;
	val[6] = 0.61721461653440386;
	val[7] = 0.61834711965622806;
	val[8] = 0.61791440652881789;
	val[9] = 0.61807966846989581;

	// get other random values to check centered around val[0].
	double range = 0.2;
	for (i = NV; i < (numExtraV + NV); i++)
	{
		uint64_t ur = gmp_urandomb_ui(gmprand, 52);
		double r = (val[0] - range / 2.0) +
			((double)ur * 2.2204460492503130808472633361816e-16) * range;
		val[i] = r;
	}

	i = 0;
	q = plist[i];
	trad_cost = 0.;

	while (q < B1)
	{
		i++;
		q = plist[i];
	}
	nump = i;

	v = (double*)malloc(nump * sizeof(double));
	pcost = (double*)malloc(nump * sizeof(double));

	printf("maxp is %d, %d primes\n", plist[nump - 1], nump);
	pshuf = (int*)malloc(nump * sizeof(int));
	mpz_set_ui(maxn, nump - 1);

	pcost[0] = 5.0;
	total_cost = 0.0;

	for (i = 1; i < nump; i++)
	{
		q = plist[i];
		for (d = 0, cmin = ADD * (double)q; d < (numExtraV + NV); d++)
		{
			cost = lucas_cost(q, val[d]);
			if (cost < cmin)
			{
				cmin = cost;
				best_index = d;
			}
		}

		pcost[i] = cmin;
		v[i] = val[best_index];
		total_cost += cmin;
	}

	// to complete the cost evaluation for the given B1, add in
	// the contributions of 2 and any powers of small primes p that
	// "fit" in B1 beyond the 1 copy already included in the cost.
	trad_cost = total_cost;
	q = 2;
	while (q < B1)
	{
		total_cost += 5;
		q *= 2;
	}

	for (i = 1; i < nump; i++)
	{
		q = plist[i] * plist[i];
		j = 0;
		cmin = 0.0;
		while (q < B1)
		{
			cmin += pcost[i];
			q *= plist[i];
			j++;
		}
		if (j > 0)
			total_cost += cmin;
		else
			break;
	}

	printf("total traditional (optimized) B1 cost (up to prime %d index %u) is %1.0f\n", 
		plist[nump - 1], nump, total_cost);
	trad_cost = total_cost - trad_cost;
	printf("small prime powers add %1.0f to the base cost\n", trad_cost);

	int seed;
	double lowest_cost = 999999999.9;
	for (seed = 42; seed < numSeeds; seed++)
	{
		//printf("seed %d\n");
		gmp_randseed_ui(gmprand, seed);

		// shuffle the primes > 2 for this seed
		for (i = 0; i < nump; i++)
		{
			pshuf[i] = i;
		}
		for (i = 1; i < nump; i++)
		{
			mpz_urandomm(rndn, gmprand, maxn);
			int idx = mpz_get_ui(rndn);
			int swp = pshuf[i];
			if (idx == 0) idx = 1;
			pshuf[i] = pshuf[idx];
			pshuf[idx] = swp;
		}

		// for each potential group of 2 or 3 primes, see
		// if a lower cost exists for some V compared to the
		// base cost of the primes summed individually.
		// expand:
		// for each group of 4 primes, check the following cases
		// and select the cheapest:
		
		total_cost = 0.0;
		int pairings = 0;
		int triples = 0;
		int quads = 0;
		
#if 1
		num_groups = 0;
		for (i = 1; i < nump - 3; )
		{
			int found = 0;
			q = 1;
			double base_cost = 0.0;
			for (j = 0; (j < 2) && ((i + 2) < nump); j++)
			{
				q *= plist[pshuf[i + j]];
				base_cost += pcost[pshuf[i + j]];
			}

			for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
			{
				cost = lucas_cost(q, val[d]);
				if (cost < cmin)
				{
					cmin = cost;
					best_index = d;
				}
			}

			if (cmin < base_cost)
			{
				grp_idx[num_groups] = i;
				grp_sz[num_groups] = 2;
				grp_cost[num_groups] = cmin;
				num_groups++;
				total_cost += cmin;
				pairings++;
				i += 2;
				found = 1;
			}
			else
			{
				// try to find beneficial triples
				q = 1;
				base_cost = 0.0;
				for (j = 0; (j < 3) && ((i + 3) < nump); j++)
				{
					q *= plist[pshuf[i + j]];
					base_cost += pcost[pshuf[i + j]];
				}

				for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
				{
					cost = lucas_cost(q, val[d]);
					if (cost < cmin)
					{
						cmin = cost;
						best_index = d;
					}
				}

				if (cmin < base_cost)
				{
					grp_idx[num_groups] = i;
					grp_sz[num_groups] = 3;
					grp_cost[num_groups] = cmin;
					num_groups++;
					total_cost += cmin;
					triples++;
					i += 3;
					found = 1;
				}
			}

			if (found == 0)
			{
				total_cost += pcost[pshuf[i]];
				i++;
			}
		}

#elif 0
		for (i = 0; i < nump - 4; )
		{
			double costs[15];

			for (j = 0; j < 14; j++)
			{
				switch (j)
				{
				case 0:
					// 00: p0 + p1 + p2 + p3; i++
					costs[j] = pcost[pshuf[i]] + pcost[pshuf[i + 1]] + pcost[pshuf[i + 2]] + pcost[pshuf[i + 3]];
					break;
				case 1:
					// 01: p0 * p1 + p2 + p3; i+=2
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 2]] + pcost[pshuf[i + 3]];
					break;
				case 2:
					// 02: p0 * p2 + p1 + p3; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 1]] + pcost[pshuf[i + 3]];
					break;
				case 3:
					// 03: p0 * p3 + p1 + p2; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 1]] + pcost[pshuf[i + 2]];
					break;
				case 4:
					// 04: p1 * p2 + p0 + p3; i+=4
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 0]] + pcost[pshuf[i + 3]];
					break;
				case 5:
					// 05: p1 * p3 + p0 + p2; i+=4
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 0]] + pcost[pshuf[i + 2]];
					break;
				case 6:
					// 06: p2 * p3 + p0 + p1; i+=4
					q = plist[pshuf[i + 2]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 0]] + pcost[pshuf[i + 1]];
					break;
				case 7:
					// 07: p0 * p1 + p2 * p3; i+=2
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin;
					q = plist[pshuf[i + 2]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] += cmin;
					break;
				case 8:
					// 08: p0 * p2 + p1 * p3; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin;
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] += cmin;
					break;
				case 9:
					// 09: p0 * p3 + p1 * p2; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin;
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] += cmin;
					break;
				case 10:
					// 10: p0 * p1 * p2 + p3; i+=3
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 3]];
					break;
				case 11:
					// 11: p0 * p1 * p3 + p2; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 2]];
					break;
				case 12:
					// 12: p0 * p2 * p3 + p1; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 2]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 1]];
					break;
				case 13:
					// 13: p1 * p2 * p3 + p0; i+=4
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 2]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 0]];
					break;
				case 14:
					// 14: p0 * p1 * p2 * p3; i+=4
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]] * plist[pshuf[i + 2]] * plist[pshuf[i + 3]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin;
					break;
				}
			}

			cmin = costs[0];
			best_index = 0;
			for (j = 0; j < 14; j++)
			{
				if (costs[j] < cmin)
				{
					best_index = j;
					cmin = costs[j];
				}
			}

			switch (best_index)
			{
			case 0:  total_cost += pcost[pshuf[i]]; i++; break;
			case 1:  total_cost += costs[best_index] - pcost[pshuf[i + 2]] - pcost[pshuf[i + 3]]; i += 2; pairings++; break;
			case 2:  total_cost += costs[best_index]; i += 4; pairings++; break;
			case 3:  total_cost += costs[best_index]; i += 4; pairings++; break;
			case 4:  total_cost += costs[best_index]; i += 4; pairings++; break;
			case 5:  total_cost += costs[best_index]; i += 4; pairings++; break;
			case 6:  total_cost += costs[best_index]; i += 4; pairings++; break;
			case 7:  total_cost += costs[best_index]; i += 4; pairings += 2; break;
			case 8:  total_cost += costs[best_index]; i += 4; pairings += 2; break;
			case 9:  total_cost += costs[best_index]; i += 4; pairings += 2; break;
			case 10: total_cost += costs[best_index]; i += 4; triples++; break;
			case 11: total_cost += costs[best_index]; i += 4; triples++; break;
			case 12: total_cost += costs[best_index]; i += 4; triples++; break;
			case 13: total_cost += costs[best_index]; i += 4; triples++; break;
			case 14: total_cost += costs[best_index]; i += 4; quads++;  break;
			}

		}
#else

		num_groups = 0;
		for (i = 1; i < nump - 3; )
		{
			double costs[5];

			for (j = 0; j < 5; j++)
			{
				switch (j)
				{
				case 0:
					// 00: p0 + p1 + p2; i++
					costs[j] = pcost[pshuf[i]] + pcost[pshuf[i + 1]] + pcost[pshuf[i + 2]];
					break;
				case 1:
					// 01: p0 * p1 + p2; i+=2
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 2]];
					break;
				case 2:
					// 02: p0 * p2 + p1; i+=3
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 1]];
					break;
				case 3:
					// 03: p1 * p2 + p0; i+=3
					q = plist[pshuf[i + 1]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin + pcost[pshuf[i + 0]];
					break;
				case 4:
					// 10: p0 * p1 * p2; i+=3
					q = plist[pshuf[i + 0]] * plist[pshuf[i + 1]] * plist[pshuf[i + 2]];
					for (d = 0, cmin = ADD * (double)q; d < (numExtraV / 100 + NV); d++)
					{
						cost = lucas_cost(q, val[d]);
						if (cost < cmin)
						{
							cmin = cost;
							best_index = d;
						}
					}
					costs[j] = cmin;
					break;
				}
			}

			cmin = costs[0];
			best_index = 0;
			for (j = 1; j < 5; j++)
			{
				if (costs[j] < cmin)
				{
					best_index = j;
					cmin = costs[j];
				}
			}

			switch (best_index)
			{
			case 0:  total_cost += pcost[pshuf[i]]; i++; break;
			case 1:  
				total_cost += costs[best_index] - pcost[pshuf[i + 2]]; 
				grp_idx[num_groups] = i;
				grp_sz[num_groups] = 2;
				grp_cost[num_groups] = costs[best_index] - pcost[pshuf[i + 2]];
				num_groups++;
				i += 2; 
				pairings++; 
				break;
			case 2:  
				total_cost += costs[best_index]; 
				grp_idx[num_groups] = i;
				grp_sz[num_groups] = 4;
				grp_cost[num_groups] = costs[best_index] - pcost[pshuf[i + 1]];
				num_groups++;
				i += 3; 
				pairings++; 
				break;
			case 3:  
				total_cost += costs[best_index]; 
				grp_idx[num_groups] = i + 1;
				grp_sz[num_groups] = 2;
				grp_cost[num_groups] = costs[best_index] - pcost[pshuf[i + 0]];
				num_groups++;
				i += 3; 
				pairings++; 
				break;
			case 4:  
				total_cost += costs[best_index]; 
				grp_idx[num_groups] = i;
				grp_sz[num_groups] = 3;
				grp_cost[num_groups] = costs[best_index];
				num_groups++;
				i += 3; 
				triples++; 
				break;
			}

		}
#endif

		for (; i < nump; i++)
		{
			total_cost += pcost[pshuf[i]];
		}

		// find lowest cost seed
		if (total_cost < lowest_cost)
		{
			printf("seed %d has new lowest cost %1.0f (%d pairings, %d triples, %d quads)\n", 
				seed, total_cost + trad_cost, pairings, triples, quads);
			for (i = 0; i < num_groups; i++)
			{
				if (grp_sz[i] == 2)
				{
					printf("group %d: p = [%d,%d] cost = [%1.0f + %1.0f = %1.0f], grpcost = %1.0f\n",
						i, plist[pshuf[grp_idx[i]]], plist[pshuf[grp_idx[i] + 1]],
						pcost[pshuf[grp_idx[i]]], pcost[pshuf[grp_idx[i] + 1]],
						pcost[pshuf[grp_idx[i]]] + pcost[pshuf[grp_idx[i] + 1]],
						grp_cost[i]);
				}
				else if (grp_sz[i] == 3)
				{
					printf("group %d: p = [%d,%d,%d] cost = [%1.0f + %1.0f + %1.0f = %1.0f], grpcost = %1.0f\n",
						i, plist[pshuf[grp_idx[i]]], plist[pshuf[grp_idx[i] + 1]], plist[pshuf[grp_idx[i] + 2]],
						pcost[pshuf[grp_idx[i]]], pcost[pshuf[grp_idx[i] + 1]], pcost[pshuf[grp_idx[i] + 2]],
						pcost[pshuf[grp_idx[i]]] + pcost[pshuf[grp_idx[i] + 1]] + pcost[pshuf[grp_idx[i] + 2]],
						grp_cost[i]);
				}
				else if (grp_sz[i] == 4)
				{
					printf("group %d: p = [%d,%d] cost = [%1.0f + %1.0f = %1.0f], grpcost = %1.0f\n",
						i, plist[pshuf[grp_idx[i]]], plist[pshuf[grp_idx[i] + 2]],
						pcost[pshuf[grp_idx[i]]], pcost[pshuf[grp_idx[i] + 2]],
						pcost[pshuf[grp_idx[i]]] + pcost[pshuf[grp_idx[i] + 2]],
						grp_cost[i]);
				}
			}
			lowest_cost = total_cost;
		}

	}

	mpz_clear(rndn);
	mpz_clear(maxn);
	exit(0);
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

		gpu_cofactor_ctx->lpba = 31;
		gpu_cofactor_ctx->lpbr = 31;
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
			uint16_t lpb[2] = { 31, 31 };
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

			mpz_set_ui(large_primes1[0], 0);

			if (mpz_sgn(large_factors[1]) < 0)
			{
				mpz_neg(large_factors[1], large_factors[1]);
				tinypm1(large_factors[1], large_primes1[0], 333, 0);

				if (mpz_cmp_ui(large_primes1[0], 1) > 0)
				{
					//gmp_printf("found factor %Zd of %Zd with P-1\n",
					//	large_primes1[0], large_factors[1]);
					rb->num_success++;
				}
			}

			continue;
			
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

int meas_prac(uint64_t c, double v)
{
	uint64_t d, e, r;
	int print_seq = 0;
	int print_vals = 0;
	int tmp;
	int print_strs = 0;

	// we require c != 0
	int shift = 0;
	while ((c & 1) == 0)
	{
		c >>= 1;
		shift++;
	}

	d = c;
	r = (uint64_t)((double)d * v + 0.5);

	d = c - r;
	e = 2 * r - c;

	int p1val = 1; 
	int p2val = 1;
	int p3val = 1;
	int p4val = 0;

	char p1str[8];
	char p2str[8];
	char p3str[8];
	char p4str[8];
	char tstr[8];

	strcpy(p1str, "p1");
	strcpy(p2str, "p2");
	strcpy(p3str, "p3");
	strcpy(p4str, "p4");

	// the first one is always a doubling
	// point1 is [1]P
	//pt1.X[0] = pt2.X[0] = pt3.X[0] = P->X[0];
	//pt1.Z[0] = pt2.Z[0] = pt3.Z[0] = P->Z[0];
	//pt1.X[1] = pt2.X[1] = pt3.X[1] = P->X[1];
	//pt1.Z[1] = pt2.Z[1] = pt3.Z[1] = P->Z[1];
	//pt1.X[2] = pt2.X[2] = pt3.X[2] = P->X[2];
	//pt1.Z[2] = pt2.Z[2] = pt3.Z[2] = P->Z[2];
	//
	//modsub96(pt1.X, pt1.Z, d1, n);
	//modadd96(pt1.X, pt1.Z, s1, n);

	// point2 is [2]P
	//udup96(s, rho, n, s1, d1, &pt1);
	if (print_strs) printf("p1.X[0] = p2.X[0] = p3.X[0] = P->X[0];\n");
	if (print_strs) printf("p1.Z[0] = p2.Z[0] = p3.Z[0] = P->Z[0];\n");
	if (print_strs) printf("p1.X[1] = p2.X[1] = p3.X[1] = P->X[1];\n");
	if (print_strs) printf("p1.Z[1] = p2.Z[1] = p3.Z[1] = P->Z[1];\n");
	if (print_strs) printf("p1.X[2] = p2.X[2] = p3.X[2] = P->X[2];\n");
	if (print_strs) printf("p1.Z[2] = p2.Z[2] = p3.Z[2] = P->Z[2];\n");
	if (print_strs) printf("udup96as(s, rho, n, &%s); // %d = 2 * %d\n",
		p1str, p1val * 2, p1val);

	int cost = DUP;
	int seqnum = 0;
	int swaps = 0;
	//printf("sequence for %06lu:\n", c);

	p1val = 2;

	while (d != e)
	{
		if (d < e)
		{
			r = d;
			d = e;
			e = r;
			//swap96(pt1.X, pt2.X);
			//swap96(pt1.Z, pt2.Z);
			tmp = p1val;
			p1val = p2val;
			p2val = tmp;

			strcpy(tstr, p1str);
			strcpy(p1str, p2str);
			strcpy(p2str, tstr);

			if (print_seq) printf("0,");
			//if (print_vals) printf("0: %d,%d,%d,%d\n", p1val, p2val, p3val, p4val);
			seqnum++;
			swaps += 2;
		}
		if (d - e <= e / 4 && ((d + e) % 3) == 0)
		{
			d = (2 * d - e) / 3;
			e = (e - d) / 2;

			//uadd96(rho, n, pt1, pt2, pt3, &pt4); // T = A + B (C)
			//uadd96(rho, n, pt4, pt1, pt2, &pt5); // T2 = T + A (B)
			//uadd96(rho, n, pt2, pt4, pt1, &pt2); // B = B + T (A)
			//
			//swap96(pt1.X, pt5.X);
			//swap96(pt1.Z, pt5.Z);
			printf("1,"); seqnum++;
			cost += 18;
		}
		else if (d - e <= e / 4 && (d - e) % 6 == 0)
		{
			d = (d - e) / 2;

			//modsub96(pt1.X, pt1.Z, d1, n);
			//modadd96(pt1.X, pt1.Z, s1, n);
			//
			//uadd96(rho, n, pt1, pt2, pt3, &pt2);        // B = A + B (C)
			//udup96(s, rho, n, s1, d1, &pt1);        // A = 2A
			printf("2,"); seqnum++;
			cost += 11;
		}
		else if ((d + 3) / 4 <= e)
		{
			d -= e;

			//uadd96(rho, n, pt2, pt1, pt3, &pt4);        // T = B + A (C)
			
			//threeswap96(pt2.X, pt4.X, pt3.X);		// 2 <-- 4, 4 <-- 3, 3 <-- 2
			//threeswap96(pt2.Z, pt4.Z, pt3.Z);
			p4val = p2val + p1val;
			if (print_vals) printf("3: [p4]%d = [p2]%d + [p1]%d ([p3]%d)\n",
				p4val, p2val, p1val, p3val);
			if (print_strs) printf("uadd96(rho, n, %s, %s, %s, &%s); // %d = %d + %d (%d)\n",
				p2str, p1str, p3str, p4str, p4val, p2val, p1val, p3val);

			tmp = p2val;
			p2val = p4val;
			p4val = p3val;
			p3val = tmp;

			strcpy(tstr, p2str);
			strcpy(p2str, p4str);
			strcpy(p4str, p3str);
			strcpy(p3str, tstr);

			//if (print_vals) printf("3: %d,%d,%d,%d\n", p1val, p2val, p3val, p4val);
			if (print_seq) printf("3,");
			swaps += 2;
			seqnum++;
			cost += ADD;
		}
		else if ((d + e) % 2 == 0)
		{
			d = (d - e) / 2;

			//modsub96(pt1.X, pt1.Z, d2, n);
			//modadd96(pt1.X, pt1.Z, s2, n);
			//
			//uadd96(rho, n, pt2, pt1, pt3, &pt2);        // B = B + A (C)
			tmp = p2val + p1val;
			if (print_vals) printf("4: [p2]%d = [p2]%d + [p1]%d ([p3]%d)\n", tmp, p2val, p1val, p3val);
			if (print_strs) printf("uadd96(rho, n, %s, %s, %s, &%s); // %d = %d + %d (%d)\n",
				p2str, p1str, p3str, p2str, tmp, p2val, p1val, p3val);
			p2val = tmp;
			//udup96(s, rho, n, s2, d2, &pt1);        // A = 2A
			if (print_vals) printf("4: [p1]%d = 2 * [p1]%d\n", p1val * 2, p1val);
			if (print_strs) printf("udup96as(s, rho, n, &%s); // %d = 2 * %d\n",
				p1str, p1val * 2, p1val);
			p1val *= 2;
			if (print_seq) printf("4,"); 
			//if (print_vals) printf("4: %d,%d,%d,%d\n", p1val, p2val, p3val, p4val);
			seqnum++;
			cost += (ADD + DUP);
		}
		else if (d % 2 == 0)
		{
			d /= 2;

			//modsub96(pt1.X, pt1.Z, d2, n);
			//modadd96(pt1.X, pt1.Z, s2, n);
			//
			//uadd96(rho, n, pt3, pt1, pt2, &pt3);        // C = C + A (B)
			//udup96(s, rho, n, s2, d2, &pt1);        // A = 2A
			printf("5,"); seqnum++;
			cost += 11;
		}
		else if (d % 3 == 0)
		{
			d = d / 3 - e;

			//modsub96(pt1.X, pt1.Z, d1, n);
			//modadd96(pt1.X, pt1.Z, s1, n);
			//
			//udup96(s, rho, n, s1, d1, &pt4);        // T = 2A
			//uadd96(rho, n, pt1, pt2, pt3, &pt5);        // T2 = A + B (C)
			//uadd96(rho, n, pt4, pt1, pt1, &pt1);        // A = T + A (A)
			//uadd96(rho, n, pt4, pt5, pt3, &pt4);        // T = T + T2 (C)
			//
			//threeswap96(pt3.X, pt2.X, pt4.X);
			//threeswap96(pt3.Z, pt2.Z, pt4.Z);
			printf("6,"); seqnum++;
			cost += 23;
		}
		else if ((d + e) % 3 == 0)
		{
			d = (d - 2 * e) / 3;

			//uadd96(rho, n, pt1, pt2, pt3, &pt4);        // T = A + B (C)
			//
			//modsub96(pt1.X, pt1.Z, d2, n);
			//modadd96(pt1.X, pt1.Z, s2, n);
			//uadd96(rho, n, pt4, pt1, pt2, &pt2);        // B = T + A (B)
			//udup96(s, rho, n, s2, d2, &pt4);        // T = 2A
			//uadd96(rho, n, pt1, pt4, pt1, &pt1);        // A = A + T (A) = 3A
			printf("7,"); seqnum++;
			cost += 23;
		}
		else if ((d - e) % 3 == 0)
		{
			d = (d - e) / 3;

			//uadd96(rho, n, pt1, pt2, pt3, &pt4);        // T = A + B (C)
			//
			//modsub96(pt1.X, pt1.Z, d2, n);
			//modadd96(pt1.X, pt1.Z, s2, n);
			//uadd96(rho, n, pt3, pt1, pt2, &pt3);        // C = C + A (B)
			//
			//swap96(pt2.X, pt4.X);
			//swap96(pt2.Z, pt4.Z);
			//
			//udup96(s, rho, n, s2, d2, &pt4);        // T = 2A
			//uadd96(rho, n, pt1, pt4, pt1, &pt1);        // A = A + T (A) = 3A
			printf("8,"); seqnum++;
			cost += 23;
		}
		else
		{
			e /= 2;

			//modsub96(pt2.X, pt2.Z, d2, n);
			//modadd96(pt2.X, pt2.Z, s2, n);
			//
			//uadd96(rho, n, pt3, pt2, pt1, &pt3);        // C = C + B (A)
			//udup96(s, rho, n, s2, d2, &pt2);        // B = 2B
			printf("9,"); seqnum++;
			cost += 11;
		}
	}

	//uadd96(rho, n, pt1, pt2, pt3, P);     // A = A + B (C)
	if (print_vals) printf("a: p[%d] = [p1]%d + [p2]%d ([p3]%d)\n", 
		p1val + p2val, p1val, p2val, p3val);
	if (print_strs) printf("uadd96(rho, n, %s, %s, %s, P); // %d = %d + %d (%d)\n",
		p1str, p2str, p3str, p1val + p2val, p1val, p2val, p3val);
	p1val = p1val + p2val;
	//if (print_vals) printf("a: %d,%d,%d,%d\n", p1val, p2val, p3val, p4val);
	if (print_seq) printf("10,"); 
	seqnum++;
	cost += ADD;

	int i;
	for (i = 0; i < shift; i++)
	{
		//modsub96(P->X, P->Z, d1, n);
		//modadd96(P->X, P->Z, s1, n);
		//udup96(s, rho, n, s1, d1, P);     // P = 2P
		if (print_seq) printf("11,"); 
		seqnum++;
		cost += DUP;
	}

	//printf("\ntotal cost for %d steps = %d + %d swaps\n", seqnum, cost, swaps);
	return cost;
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

    strcpy(fname, options->file);

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
	else if (batch_alg == 4)
	{
		// semiprime list generation, in the relations format 
		// that relation_batch_t wants.
		FILE* out;
		int i;
		mpz_t tmp1, tmp2, tmp3, tmp4;
		uint32_t num = options->curves_3lp;
		uint32_t bits = options->b1_3lp;

		gmp_randstate_t gmp_randstate;
		mpz_init(tmp1);
		mpz_init(tmp2);
		mpz_init(tmp3);
		mpz_init(tmp4);
		
		gmp_randinit_default(gmp_randstate);
		gmp_randseed_ui(gmp_randstate, options->stop_nofactor);

		out = fopen(options->file, "w");
		if (out == NULL)
		{
			printf("couldn't open %s for writing\n", options->file);
			exit(1);
		}

		for (i = 0; i < num; i++)
		{
			mpz_urandomb(tmp3, gmp_randstate, bits / 2);
			mpz_setbit(tmp3, bits / 2 - 1);
			mpz_nextprime(tmp1, tmp3);
			if (bits & 1)
			{
				mpz_urandomb(tmp3, gmp_randstate, bits / 2 + 1);
				mpz_setbit(tmp3, bits / 2);
			}
			else
			{
				mpz_urandomb(tmp3, gmp_randstate, bits / 2);
				mpz_setbit(tmp3, bits / 2 - 1);
			}
			mpz_nextprime(tmp2, tmp3);
			mpz_mul(tmp3, tmp2, tmp1);
			if (i == 0)
			{
				// put one 3LP number in there so it will
				// autodetect which side is 2LP correctly.
				mpz_urandomb(tmp4, gmp_randstate, 90);
				gmp_fprintf(out, "-%Zd,-%Zd:%Zd,%Zd:4,5:6,7\n",
					tmp3, tmp4, tmp1, tmp2);
			}
			else
			{
				gmp_fprintf(out, "-%Zd,1:%Zd,%Zd:4,5:6,7\n",
					tmp3, tmp1, tmp2);
			}
			
		}
		fclose(out);

		printf("generated %d semiprimes in file %s\n", num, options->file);
		mpz_clear(tmp1);
		mpz_clear(tmp2);
		mpz_clear(tmp3);
		mpz_clear(tmp4);

		return 0;
	}
	else if (batch_alg == 5)
	{
		printf("b1=%d\n", options->b1_3lp);
		int stg1 = options->b1_3lp;
		int cost = 0;
		if (stg1 > 400)
		{
			// ./cuda_3lp -b1 500 -b2 100000 -m 2
			// anything greater than 400 gets B1=500
			// pair-optimized only within the set 400 < p < 500
			cost += meas_prac(184861, 0.628445063812485882);//[401,461] is 155.0, savings = 2
			cost += meas_prac(409, 0.551390822543526449);
			cost += meas_prac(419, 0.524531838023777786);
			cost += meas_prac(421, 0.643616254685781319);
			cost += meas_prac(431, 0.551390822543526449);
			cost += meas_prac(433, 0.553431118763021646);
			cost += meas_prac(439, 0.618033988749894903);
			cost += meas_prac(217513, 0.586747080461318182);//[443,491] is 156.0, savings = 1
			cost += meas_prac(449, 0.612429949509495031);
			cost += meas_prac(457, 0.612429949509495031);
			//meas_prac(461, 0.520959819992697026);
			//[401,461] is 155.0, savings = 2
			//[43, 461] is 126.0, savings = 2
			//[367,461] is 156.0, savings = 4
			//[379,461] is 156.0, savings = 3
			cost += meas_prac(463, 0.553431118763021646);
			cost += meas_prac(467, 0.553431118763021646);
			cost += meas_prac(479, 0.618033988749894903);
			cost += meas_prac(487, 0.591965645556728037);
			//[431,491] is 155.0, savings = 1
			//[11,491] is 108.0, savings = 2
			//[443,491] is 156.0, savings = 1
			//[457,491] is 156.0, savings = 1
			//meas_prac(491, 0.618347119656228017);
			cost += meas_prac(499, 0.618033988749894903);

		}

		if (stg1 > 300)
		{
			// anything greater than 300 gets B1=400
			// pair-optimized within the set 250 < p < 400
			cost +=  meas_prac(307, 0.580178728295464130);
			cost +=  meas_prac(311, 0.618033988749894903);
			cost +=  meas_prac(313, 0.618033988749894903);
			cost +=  meas_prac(317, 0.618033988749894903);
			//meas_prac(331, 0.543797196679826511);
			cost +=  meas_prac(337, 0.618033988749894903);
			cost +=  meas_prac(124573, 0.552705982126542983); //[347,359], savings = 3
			cost +=  meas_prac(349, 0.632839806088706269);
			cost +=  meas_prac(95663, 0.618033988749894903);// [271,353], savings = 2
			//meas_prac(359, 0.612429949509495031);
			cost +=  meas_prac(142763, 0.524817056539543136);// [367,389], savings = 1
			cost +=  meas_prac(373, 0.524531838023777786);
			cost +=  meas_prac(145157, 0.580178728295464130);//[379,383], savings = 2
			//meas_prac(383, 0.537965503694917802);
			//meas_prac(389, 0.632839806088706269);
			cost +=  meas_prac(397, 0.580178728295464130);
			// one more factor of 7 will fit - and it pairs with 331, savings 2
			cost +=  meas_prac(2317, 0.618033988749894903);  // 7 x 331, savings 2
		}

		if (stg1 > 250)
		{
			// anything greater than 250 gets B1=300
			// pair-optimized within the set 200 < p < 300
			cost += meas_prac(251, 0.541554796058780874);
			//meas_prac(257, 0.551390822543526449); // paired with 223 below
			cost += meas_prac(263, 0.612429949509495031);
			cost += meas_prac(269, 0.618033988749894903);
			if (stg1 > 300)
			{
				// paired with 353, above
			}
			else
			{
				cost += meas_prac(271, 0.618033988749894903);
			}
			cost += meas_prac(277, 0.618033988749894903);
			cost += meas_prac(281, 0.580178728295464130);
			cost += meas_prac(283, 0.580178728295464130);
			cost += meas_prac(293, 0.551390822543526449);
		}

		if (stg1 > 200)
		{
			// anything greater than 200 gets B1=250
			// pair-optimized only within the set 200 < p < 250
			cost += meas_prac(48319, 0.552793637425581075);     // 211 x 229 savings 3
			if (stg1 > 250)
			{
				cost += meas_prac(57311, 0.552740754023503311); // 223 x 257 savings 3
				cost += meas_prac(241, 0.625306711365725132);   // [241,347], savings = 2
			}
			else
			{
				cost += meas_prac(53743, 0.718522647487825128); // 223 x 241 savings 2
			}
			cost += meas_prac(54253, 0.580178728295464130);     // 227 x 239 savings 2
			cost += meas_prac(233, 0.618033988749894903);
			// one more factor of 3 will fit
			cost += meas_prac(3, 0.618033988749894903);
		}

		if (stg1 > 175)
		{
			// anything greater than 175 gets B1=200.
			// here we have pair-optimized some primes
			cost += meas_prac(3, 0.618033988749894903);
			cost += meas_prac(3, 0.618033988749894903);
			cost += meas_prac(3, 0.618033988749894903);
			cost += meas_prac(3, 0.618033988749894903);
			cost += meas_prac(5, 0.618033988749894903);
			cost += meas_prac(5, 0.618033988749894903);
			cost += meas_prac(5, 0.618033988749894903);
			cost += meas_prac(7, 0.618033988749894903);
			cost += meas_prac(7, 0.618033988749894903);
			cost += meas_prac(11, 0.580178728295464130);
			cost += meas_prac(11, 0.580178728295464130);
			cost += meas_prac(13, 0.618033988749894903);
			cost += meas_prac(13, 0.618033988749894903);
			cost += meas_prac(17, 0.618033988749894903);
			cost += meas_prac(19, 0.618033988749894903);
			cost += meas_prac(3427, 0.618033988749894903);
			cost += meas_prac(29, 0.548409048446403258);
			cost += meas_prac(31, 0.618033988749894903);
			cost += meas_prac(4181, 0.618033988749894903);
			cost += meas_prac(2173, 0.618033988749894903);
			cost += meas_prac(43, 0.618033988749894903);
			cost += meas_prac(47, 0.548409048446403258);
			cost += meas_prac(8909, 0.580178728295464130);
			cost += meas_prac(12017, 0.643969713705029423);
			cost += meas_prac(67, 0.580178728295464130);
			cost += meas_prac(71, 0.591965645556728037);
			cost += meas_prac(73, 0.618033988749894903);
			cost += meas_prac(79, 0.618033988749894903);
			cost += meas_prac(8051, 0.632839806088706269);
			cost += meas_prac(89, 0.618033988749894903);
			cost += meas_prac(101, 0.556250337855490828);
			cost += meas_prac(103, 0.632839806088706269);
			cost += meas_prac(107, 0.580178728295464130);
			cost += meas_prac(109, 0.548409048446403258);
			cost += meas_prac(17653, 0.586779411332316370);
			cost += meas_prac(131, 0.618033988749894903);
			cost += meas_prac(22879, 0.591384619013580526);
			cost += meas_prac(157, 0.640157392785047019);
			cost += meas_prac(163, 0.551390822543526449);
			cost += meas_prac(173, 0.612429949509495031); //[173,347], savings = 2
			cost += meas_prac(179, 0.618033988749894903); //[179,379], savings = 1, [179,461], savings = 3
			cost += meas_prac(181, 0.551390822543526449);
			cost += meas_prac(191, 0.618033988749894903);
			cost += meas_prac(193, 0.618033988749894903);
			cost += meas_prac(199, 0.551390822543526449);
		}

		int q = 2;
		while (q < stg1)
		{
			cost += 5;
			q *= 2;
		}

		printf("total cost is %d muls\n", cost);
		return 0;
	}
	else
	{
		//lucas_opt2(options->b1_3lp, options->b2_3lp);
		lucas_opt3(options->b1_3lp, options->b2_3lp, options->curves_3lp);
	}

    process_batch(&rb, lpbr, lpba, options->file, "", 1, batch_alg,
		options->b1_3lp, options->b2_3lp, options->curves_3lp, options->stop_nofactor);

    return 0;
}
