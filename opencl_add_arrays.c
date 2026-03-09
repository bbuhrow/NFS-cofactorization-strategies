/*
 * opencl_add_arrays.c
 *
 * Demonstrates how to use OpenCL to add two large arrays of 32-bit integers
 * on a GPU (or any OpenCL device).
 *
 * Compile with:
 *   Linux/macOS:  gcc opencl_add_arrays.c -o opencl_add -lOpenCL
 *   macOS only:   gcc opencl_add_arrays.c -o opencl_add -framework OpenCL
 *
 * OpenCL Concepts Covered:
 *   1. Platform & Device selection
 *   2. Context & Command Queue creation
 *   3. Writing a kernel (the GPU function)
 *   4. Compiling the kernel at runtime
 *   5. Allocating GPU memory (Buffers)
 *   6. Copying data Host → GPU
 *   7. Launching the kernel
 *   8. Copying results GPU → Host
 *   9. Cleanup
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>   // macOS ships OpenCL as a framework
#else
  #include <CL/cl.h>           // Linux / Windows: install opencl-headers
#endif

/* -----------------------------------------------------------------------
 * THE KERNEL SOURCE
 *
 * This small C-like program runs on the GPU.
 * "__kernel"  marks it as an entry point callable from the host.
 * "__global"  means the pointer lives in global GPU memory (the big pool).
 * "get_global_id(0)" returns this work-item's unique index — think of it
 * as the loop variable 'i' in: for (int i = 0; i < N; i++)
 * Each work-item handles exactly one element, so all additions happen in
 * parallel.
 * ----------------------------------------------------------------------- */
const char *kernel_source =
"__kernel void add_arrays(__global const int* a,   \n"
"                         __global const int* b,   \n"
"                         __global       int* c,   \n"
"                         const int n)             \n"
"{                                                  \n"
"    int i = get_global_id(0);                      \n"
"    if (i < n)        /* bounds guard */           \n"
"        c[i] = a[i] + b[i];                        \n"
"}                                                  \n";

/* Simple error-check helper — prints a message and exits on CL error */
static void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d at: %s\n", err, msg);
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* 0. Problem size                                                      */
    /* ------------------------------------------------------------------ */
    const int N = 1 << 20;          // 1 048 576 elements (~4 MB per array)
    size_t bytes = N * sizeof(int);

    /* ------------------------------------------------------------------ */
    /* 1. Allocate and initialise host (CPU) arrays                        */
    /* ------------------------------------------------------------------ */
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);   // results land here

    for (int i = 0; i < N; i++) {
        h_a[i] = i;           // 0, 1, 2, …
        h_b[i] = N - i;       // N, N-1, N-2, …  → sum should always be N
    }

    /* ------------------------------------------------------------------ */
    /* 2. Discover OpenCL platforms and pick the first one                 */
    /*                                                                      */
    /*   A "platform" is a vendor's OpenCL implementation, e.g.:           */
    /*     - NVIDIA CUDA platform                                           */
    /*     - Intel OpenCL                                                   */
    /*     - AMD ROCm / APP SDK                                             */
    /*     - Apple's built-in (macOS)                                       */
    /* ------------------------------------------------------------------ */
    cl_uint num_platforms;
    check(clGetPlatformIDs(0, NULL, &num_platforms), "count platforms");

    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found. "
                        "Is a GPU driver/runtime installed?\n");
        return EXIT_FAILURE;
    }

    cl_platform_id *platforms =
        (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    check(clGetPlatformIDs(num_platforms, platforms, NULL), "get platforms");

    cl_platform_id platform = platforms[0];   // use first platform

    /* Print platform name for information */
    char pname[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
    printf("Platform : %s\n", pname);

    /* ------------------------------------------------------------------ */
    /* 3. Select a device                                                   */
    /*                                                                      */
    /*   We ask for a GPU first; fall back to any device (CPU/accelerator). */
    /*   A "device" is the actual compute unit: a GPU chip, a CPU, etc.    */
    /* ------------------------------------------------------------------ */
    cl_device_id device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("No GPU found, falling back to CL_DEVICE_TYPE_ALL\n");
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL),
              "get device");
    } else {
        check(err, "get GPU device");
    }

    char dname[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("Device   : %s\n", dname);

    /* ------------------------------------------------------------------ */
    /* 4. Create a Context                                                  */
    /*                                                                      */
    /*   A context groups one or more devices and manages memory objects    */
    /*   and command queues associated with them.                           */
    /* ------------------------------------------------------------------ */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "create context");

    /* ------------------------------------------------------------------ */
    /* 5. Create a Command Queue                                            */
    /*                                                                      */
    /*   Commands (memory transfers, kernel launches) are submitted to a   */
    /*   queue and executed in order on the device.                         */
    /* ------------------------------------------------------------------ */
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "create command queue");

    /* ------------------------------------------------------------------ */
    /* 6. Compile the kernel at runtime                                     */
    /*                                                                      */
    /*   OpenCL compiles kernel source on-the-fly for the target device.   */
    /*   clCreateProgramWithSource → clBuildProgram → clCreateKernel        */
    /* ------------------------------------------------------------------ */
    cl_program program =
        clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check(err, "create program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Retrieve and print build log to see compiler errors */
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, "add_arrays", &err);
    check(err, "create kernel");

    /* ------------------------------------------------------------------ */
    /* 7. Allocate GPU memory buffers                                       */
    /*                                                                      */
    /*   clCreateBuffer allocates memory on the device.                    */
    /*   CL_MEM_READ_ONLY  — device only reads this buffer                 */
    /*   CL_MEM_WRITE_ONLY — device only writes this buffer                */
    /* ------------------------------------------------------------------ */
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, NULL, &err);
    check(err, "create buffer a");
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, NULL, &err);
    check(err, "create buffer b");
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    check(err, "create buffer c");

    /* ------------------------------------------------------------------ */
    /* 8. Copy input data from Host → Device (enqueue write)               */
    /* ------------------------------------------------------------------ */
    check(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0,NULL,NULL),
          "write buffer a");
    check(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0,NULL,NULL),
          "write buffer b");
    /* CL_TRUE = blocking write: call waits until the copy is complete */

    /* ------------------------------------------------------------------ */
    /* 9. Set kernel arguments                                              */
    /*                                                                      */
    /*   Each argument corresponds to a parameter in the kernel signature: */
    /*   add_arrays(a, b, c, n)                                            */
    /* ------------------------------------------------------------------ */
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a), "arg 0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b), "arg 1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c), "arg 2");
    check(clSetKernelArg(kernel, 3, sizeof(int),    &N  ), "arg 3");

    /* ------------------------------------------------------------------ */
    /* 10. Launch the kernel                                                */
    /*                                                                      */
    /*   global_size = total number of work-items (threads) = N            */
    /*   local_size  = work-items per work-group (like a CUDA thread block) */
    /*                 NULL lets the runtime choose automatically.          */
    /*                                                                      */
    /*   The GPU will run N instances of add_arrays in parallel.           */
    /* ------------------------------------------------------------------ */
    size_t global_size = (size_t)N;
    check(clEnqueueNDRangeKernel(queue, kernel,
                                 1,           /* 1-dimensional problem */
                                 NULL,        /* global offset = 0     */
                                 &global_size,
                                 NULL,        /* local size: auto      */
                                 0, NULL, NULL),
          "enqueue kernel");

    /* Wait for the kernel to finish before reading results */
    clFinish(queue);

    /* ------------------------------------------------------------------ */
    /* 11. Copy results from Device → Host                                 */
    /* ------------------------------------------------------------------ */
    check(clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0,NULL,NULL),
          "read buffer c");

    /* ------------------------------------------------------------------ */
    /* 12. Verify results                                                   */
    /* ------------------------------------------------------------------ */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != N) {          // every element should equal N
            printf("MISMATCH at [%d]: got %d, expected %d\n", i, h_c[i], N);
            if (++errors > 10) break;
        }
    }
    if (errors == 0)
        printf("Verification PASSED — all %d sums equal %d\n", N, N);

    /* ------------------------------------------------------------------ */
    /* 13. Cleanup — release OpenCL objects in reverse creation order      */
    /* ------------------------------------------------------------------ */
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_a); free(h_b); free(h_c);
    free(platforms);

    return EXIT_SUCCESS;
}
