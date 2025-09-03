#include "cpu.h"
#include "cubiomes.h"

#include <optional>
#include <chrono>

std::optional<CpuOutput> process(Cubiomes *cubiomes, const GpuOutput &input) {
    constexpr int32_t small_biomes_pos_div = large_biomes ? 1 : 4;
    constexpr int32_t small_biomes_area_div = small_biomes_pos_div * small_biomes_pos_div;

    // return {{ input.seed, input.x, input.z, 0 }};

    cubiomes_apply_seed(cubiomes, input.seed);

    int32_t range = 40000 / small_biomes_pos_div;
    int32_t min_area = (large_biomes ? 92'000'000 : 80'000'000) / small_biomes_area_div;

    if (!cubiomes_test_monte_carlo(cubiomes, input.x, input.z, range, (double)min_area / (range * range), 0.999)) {
        // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " failed monteCarloBiomes\n", input.seed, input.x, input.z);
        return {};
    }

    min_area = 80'000'000 / small_biomes_area_div;
    PosArea res;
    if (!cubiomes_test_biome_centers(cubiomes, input.x, input.z, range, min_area, &res)) {
        // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " failed getBiomeCenters\n", input.seed, input.x, input.z);
        return {};
    }

    // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " passed\n", input.seed, input.x, input.z);
    return {{ input.seed, res.x, res.z, res.area }};
}

CpuThread::CpuThread(int id, GpuOutputs &inputs, CpuOutputs &outputs) : Thread(), id(id), inputs(inputs), outputs(outputs) {
    start();
}

void CpuThread::run() {
    std::printf("Started cpu thread %d\n", id);

    Cubiomes *cubiomes = cubiomes_create(large_biomes);

    while (!should_stop()) {
        GpuOutput input;
        {
            std::unique_lock lock(inputs.mutex);
            if (inputs.queue.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            input = inputs.queue.front();
            inputs.queue.pop();
        }

        const auto start = std::chrono::steady_clock::now();

        const auto output = process(cubiomes, input);

        const auto end = std::chrono::steady_clock::now();
        double time_total = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
        std::printf("Cpu test took %.3f s\n", time_total);

        if (!output) continue;

        {
            std::lock_guard lock(outputs.mutex);
            outputs.queue.push(output.value());
        }
    }

    cubiomes_free(cubiomes);
}