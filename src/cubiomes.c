#include "cubiomes.h"

#include "../cubiomes/finders.h"
#include <stdio.h>
#include <stdlib.h>

struct Cubiomes {
    Generator g;
};

Cubiomes *cubiomes_create(int large_biomes) {
    Cubiomes *cubiomes = malloc(sizeof(Cubiomes));
    if (cubiomes == NULL) {
        fprintf(stderr, "cubiomes_create failed\n");
        abort();
    }
    setupGenerator(&cubiomes->g, MC_NEWEST, large_biomes ? LARGE_BIOMES : 0);
    return cubiomes;
}

void cubiomes_free(Cubiomes *cubiomes) {
    free(cubiomes);
}

void cubiomes_apply_seed(Cubiomes *cubiomes, uint64_t seed) {
    applySeed(&cubiomes->g, DIM_OVERWORLD, seed);
}

static int eval(Generator *g, int scale, int x, int y, int z, void *data) {
    return sampleBiomeNoise(&g->bn, NULL, x, y, z, NULL, 0) == mushroom_fields;
}

static Range make_range_s4(int32_t x, int32_t z, int32_t range) {
    return (Range){
        .scale = 4,
        .x = (x - range / 2) / 4,
        .z = (z - range / 2) / 4,
        .sx = range / 4,
        .sz = range / 4,
        .y = 256 / 4,
        .sy = 1
    };
}

int cubiomes_test_monte_carlo(Cubiomes *cubiomes, int32_t x, int32_t z, int32_t range, double fraction, double confidence) {
    Range r = make_range_s4(x, z, range);
    uint64_t rng = cubiomes->g.seed;
    return monteCarloBiomes(&cubiomes->g, r, &rng, fraction, confidence, eval, NULL);
}

int cubiomes_test_biome_centers(Cubiomes *cubiomes, int32_t x, int32_t z, int32_t range, int32_t min_area, PosArea *out) {
    Pos pos;
    int siz;
    Range r = make_range_s4(x, z, range);
    int n = getBiomeCenters(&pos, &siz, 1, &cubiomes->g, r, mushroom_fields, min_area / 16, 1, NULL);
    if (n == 1) {
        out->x = pos.x;
        out->z = pos.z;
        out->area = siz * 16;
        return 1;
    }
    return 0;
}