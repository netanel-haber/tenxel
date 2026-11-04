/* bench.c - the speed of tenxel_classify on files, in ns, CPU cycles and instructions per byte.
 *
 * Cycles and instructions come from the CPU's counters through Linux's perf_event_open, counting user space only.
 * Without permission for them (kernel.perf_event_paranoid above 2), only the time is printed. Each file is classified
 * 40 times, and the fastest run counts. Pin it to one core for steady numbers: taskset -c 2 ./bench file.c
 *
 * Usage: bench FILE...
 */
#define main tenxel_main /* this file has its own */
#include "tenxel.c"
#undef main
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* A counter of one hardware event in this thread's user space, or -1 */
static int counter(u64 event) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_HARDWARE;
    a.config = event;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
}

static u64 count(int fd) {
    u64 v = 0;
    if (fd < 0 || read(fd, &v, sizeof v) != sizeof v)
        return 0;
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench FILE...\n");
        return 2;
    }
    int cycles = counter(PERF_COUNT_HW_CPU_CYCLES);
    int instructions = counter(PERF_COUNT_HW_INSTRUCTIONS);
    if (cycles < 0 || instructions < 0)
        fprintf(stderr, "bench: no CPU counters (see kernel.perf_event_paranoid), timing only\n");

    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) {
            perror(argv[i]);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        size_t n = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        u8 *src = malloc(n);
        u8 *cls = malloc(n);
        if (fread(src, 1, n, f) != n)
            return 1;
        fclose(f);

        double best = 1e30;
        u64 best_cycles = 0;
        u64 best_instructions = 0;
        for (int r = 0; r < 40; r++) {
            struct timespec a;
            struct timespec b;
            u64 c0 = count(cycles);
            u64 i0 = count(instructions);
            clock_gettime(CLOCK_MONOTONIC, &a);
            tenxel_classify(src, n, cls);
            clock_gettime(CLOCK_MONOTONIC, &b);
            u64 c1 = count(cycles);
            u64 i1 = count(instructions);
            double t = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
            if (t < best) {
                best = t;
                best_cycles = c1 - c0;
                best_instructions = i1 - i0;
            }
        }
        printf("%s: %zu bytes, %.3f ns/byte", argv[i], n, best / (double)n * 1e9);
        if (cycles >= 0 && instructions >= 0)
            printf(", %.2f cycles/byte, %.2f instructions/byte", (double)best_cycles / (double)n,
                   (double)best_instructions / (double)n);
        printf("\n");
        free(src);
        free(cls);
    }
    return 0;
}
