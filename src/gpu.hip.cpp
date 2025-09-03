#include "gpu.h"
#include "Random.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <array>
#include <chrono>
#include <bit>
#include <random>
#include <thread>
#include <mutex>

#define PANIC(...) { \
    std::fprintf(stderr, __VA_ARGS__); \
    std::abort(); \
}

#define TRY_HIP(expr) try_hip(expr, __FILE__, __LINE__)

void try_hip(hipError_t error, const char *file, uint64_t line) {
    if (error == hipSuccess) return;

    PANIC("HIP error at %s:%" PRIu64 ": %s\n", file, line, hipGetErrorString(error));
}

// from cubiomes
constexpr XrsrForkHash hash_continentalness { 0x83886c9d0ae3a662, 0xafa638a61b42e8ad }; // md5 "minecraft:continentalness"
constexpr XrsrForkHash hash_continentalness_large { 0x9a3f51a113fce8dc, 0xee2dbd157e5dcdad }; // md5 "minecraft:continentalness_large"
constexpr XrsrForkHash hash_octave[] {
    { 0xb198de63a8012672, 0x7b84cad43ef7b5a8 }, // md5 "octave_-12"
    { 0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8 }, // md5 "octave_-11"
    { 0x36d326eed40efeb2, 0x5be9ce18223c636a }, // md5 "octave_-10"
    { 0x082fe255f8be6631, 0x4e96119e22dedc81 }, // md5 "octave_-9"
    { 0x0ef68ec68504005e, 0x48b6bf93a2789640 }, // md5 "octave_-8"
    { 0xf11268128982754f, 0x257a1d670430b0aa }, // md5 "octave_-7"
    { 0xe51c98ce7d1de664, 0x5f9478a733040c45 }, // md5 "octave_-6"
    { 0x6d7b49e7e429850a, 0x2e3063c622a24777 }, // md5 "octave_-5"
    { 0xbd90d5377ba1b762, 0xc07317d419a7548d }, // md5 "octave_-4"
    { 0x53d39c6752dac858, 0xbcd1c5a80ab65b3e }, // md5 "octave_-3"
    { 0xb4a24d7a84e7677b, 0x023ff9668e89b5c4 }, // md5 "octave_-2"
    { 0xdffa22b534c5f608, 0xb9b67517d3665ca9 }, // md5 "octave_-1"
    { 0xd50708086cef4d7c, 0x6e1651ecc7f43309 }, // md5 "octave_0"
};

struct ImprovedNoise {
    uint8_t p[256];
    float xo;
    float yo;
    float zo;
};

struct Octave {
    ImprovedNoise noise;
    double input_factor;
    double value_factor;
};

template<size_t N>
struct NoiseParameters {
    int32_t first_octave;
    std::array<double, N> amplitudes;
};

template<size_t N>
constexpr NoiseParameters<N> make_noise_parameters(int32_t first_octave, const double (&amplitudes)[N]) {
    std::array<double, N> amp {};
    std::copy(std::begin(amplitudes), std::end(amplitudes), amp.begin());
    return { first_octave, amp };
}

constexpr auto continentalness_parameters = make_noise_parameters(-9, { 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0 });
constexpr auto continentalness_large_parameters = make_noise_parameters(-11, { 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0 });

struct OctaveConfig {
    XrsrForkHash fork_hash;
    double input_factor;
    double value_factor;
};

template<size_t N>
struct NormalNoiseConfig {
    XrsrForkHash fork_hash;
    std::array<OctaveConfig, N> octaves_a;
    std::array<OctaveConfig, N> octaves_b;
};

template<size_t N>
constexpr NormalNoiseConfig<N> make_normal_noise_config(const NoiseParameters<N> &noise_parameters, const XrsrForkHash &fork_hash) {
    NormalNoiseConfig<N> res { fork_hash };

    const auto first_octave = noise_parameters.first_octave;
    const auto &amplitudes = noise_parameters.amplitudes;

    double root_value_factor = 0.16666666666666666 / (0.1 * (1.0 + 1.0 / amplitudes.size()));

    double input_factor = 1.0 / (1 << -first_octave);
    double value_factor = (1 << (amplitudes.size() - 1)) / ((1 << amplitudes.size()) - 1.0) * root_value_factor;

    for (size_t i = 0; i < amplitudes.size(); i++) {
        res.octaves_a[i] = { hash_octave[first_octave + 12 + i], input_factor, value_factor * amplitudes[i] };
        res.octaves_b[i] = { hash_octave[first_octave + 12 + i], input_factor * 1.0181268882175227, value_factor * amplitudes[i] };
        input_factor *= 2.0;
        value_factor *= 0.5;
    }

    return res;
}

constexpr auto continentalness_config = make_normal_noise_config(continentalness_parameters, hash_continentalness);
constexpr auto continentalness_large_config = make_normal_noise_config(continentalness_large_parameters, hash_continentalness_large);
constexpr auto chosen_continentalness_config = large_biomes ? continentalness_large_config : continentalness_config;
__device__ constexpr auto device_chosen_continentalness_config = chosen_continentalness_config;

// switch - 4.745 Gsps
// int8_t[3][16] - 5.293 Gsps
// float[3][16] - 5.324 Gsps
// uint32_t[16] - 5.306 Gsps

struct GradDotTable {
    float x[16];
    float y[16];
    float z[16];
};

__device__ GradDotTable device_grad_dot_table;

void init_grad_dot_table() {
    GradDotTable table;
    table.x[ 0] =  1; table.y[ 0] =  1; table.z[ 0] =  0; // { 1,  1,  0}
    table.x[ 1] = -1; table.y[ 1] =  1; table.z[ 1] =  0; // {-1,  1,  0}
    table.x[ 2] =  1; table.y[ 2] = -1; table.z[ 2] =  0; // { 1, -1,  0}
    table.x[ 3] = -1; table.y[ 3] = -1; table.z[ 3] =  0; // {-1, -1,  0}
    table.x[ 4] =  1; table.y[ 4] =  0; table.z[ 4] =  1; // { 1,  0,  1}
    table.x[ 5] = -1; table.y[ 5] =  0; table.z[ 5] =  1; // {-1,  0,  1}
    table.x[ 6] =  1; table.y[ 6] =  0; table.z[ 6] = -1; // { 1,  0, -1}
    table.x[ 7] = -1; table.y[ 7] =  0; table.z[ 7] = -1; // {-1,  0, -1}
    table.x[ 8] =  0; table.y[ 8] =  1; table.z[ 8] =  1; // { 0,  1,  1}
    table.x[ 9] =  0; table.y[ 9] = -1; table.z[ 9] =  1; // { 0, -1,  1}
    table.x[10] =  0; table.y[10] =  1; table.z[10] = -1; // { 0,  1, -1}
    table.x[11] =  0; table.y[11] = -1; table.z[11] = -1; // { 0, -1, -1}
    table.x[12] =  1; table.y[12] =  1; table.z[12] =  0; // { 1,  1,  0}
    table.x[13] =  0; table.y[13] = -1; table.z[13] =  1; // { 0, -1,  1}
    table.x[14] = -1; table.y[14] =  1; table.z[14] =  0; // {-1,  1,  0}
    table.x[15] =  0; table.y[15] = -1; table.z[15] = -1; // { 0, -1, -1}

    void *device_grad_dot_table_addr;
    TRY_HIP(hipGetSymbolAddress((void**)&device_grad_dot_table_addr, HIP_SYMBOL(device_grad_dot_table)));
    TRY_HIP(hipMemcpy(device_grad_dot_table_addr, &table, sizeof(GradDotTable), hipMemcpyHostToDevice));
}

__device__ float gradDot(const GradDotTable &table, uint8_t p, float x, float y, float z) {
    return x * table.x[p & 0xF] + y * table.y[p & 0xF] + z * table.z[p & 0xF];
}

__device__ float smoothstep(float value) {
    return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp1(float fx, float v0, float v1) {
    return v0 + fx * (v1 - v0);
}

__device__ float lerp2(float fx, float fy, float v00, float v10, float v01, float v11) {
    return lerp1(fy, lerp1(fx, v00, v10), lerp1(fx, v01, v11));
}

__device__ float lerp3(float fx, float fy, float fz, float v000, float v100, float v010, float v110, float v001, float v101, float v011, float v111) {
    return lerp1(fz, lerp2(fx, fy, v000, v100, v010, v110), lerp2(fx, fy, v001, v101, v011, v111));
}

__device__ float sample_noise(const GradDotTable &table, const ImprovedNoise &noise, float x, float y, float z) {
    x += noise.xo;
    y += noise.yo;
    z += noise.zo;
    float floor_x = std::floor(x);
    float floor_y = std::floor(y);
    float floor_z = std::floor(z);
    float frac_x = x - floor_x;
    float frac_y = y - floor_y;
    float frac_z = z - floor_z;
    int32_t int_x = floor_x;
    int32_t int_y = floor_y;
    int32_t int_z = floor_z;
    uint8_t p0 = noise.p[(int_x    ) & 0xFF];
    uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
    uint8_t p00 = noise.p[(p0 + int_y    ) & 0xFF];
    uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
    uint8_t p10 = noise.p[(p1 + int_y    ) & 0xFF];
    uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
    float n000 = gradDot(table, noise.p[(p00 + int_z    ) & 0xFF], frac_x       , frac_y       , frac_z       );
    float n100 = gradDot(table, noise.p[(p10 + int_z    ) & 0xFF], frac_x - 1.0f, frac_y       , frac_z       );
    float n010 = gradDot(table, noise.p[(p01 + int_z    ) & 0xFF], frac_x       , frac_y - 1.0f, frac_z       );
    float n110 = gradDot(table, noise.p[(p11 + int_z    ) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z       );
    float n001 = gradDot(table, noise.p[(p00 + int_z + 1) & 0xFF], frac_x       , frac_y       , frac_z - 1.0f);
    float n101 = gradDot(table, noise.p[(p10 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y       , frac_z - 1.0f);
    float n011 = gradDot(table, noise.p[(p01 + int_z + 1) & 0xFF], frac_x       , frac_y - 1.0f, frac_z - 1.0f);
    float n111 = gradDot(table, noise.p[(p11 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
    float fx = smoothstep(frac_x);
    float fy = smoothstep(frac_y);
    float fz = smoothstep(frac_z);
    return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__device__ float wrap(float value) {
    // return value - std::floor(value / 256.0) * 256.0;
    return value;
}

template<OctaveConfig config>
__device__ float sample_octave(const GradDotTable &table, const ImprovedNoise &noise, int32_t x, int32_t y, int32_t z) {
    return sample_noise(table, noise, wrap(x * (float)config.input_factor), wrap(y * (float)config.input_factor), wrap(z * (float)config.input_factor)) * (float)config.value_factor;
}

__device__ void init_noise(ImprovedNoise &noise, XrsrRandom &&random) {
    noise.xo = random.nextFloat() * 256.0f;
    noise.yo = random.nextFloat() * 256.0f;
    noise.zo = random.nextFloat() * 256.0f;

    for (uint32_t i = 0; i < 256; i++) {
        noise.p[i] = i;
    }
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t j = random.nextInt(256 - i);
        uint8_t b = noise.p[i];
        noise.p[i] = noise.p[i + j];
        noise.p[i + j] = b;
    }
}

namespace KernelSeed1 {
    constexpr uint32_t threads_per_run = UINT64_C(1) << 14;
    constexpr uint32_t threads_per_block = 32;

    struct Result {
        ImprovedNoise continentalness_0A;
        ImprovedNoise continentalness_0B;
        ImprovedNoise continentalness_1A;
        ImprovedNoise continentalness_1B;
        ImprovedNoise continentalness_2A;
        ImprovedNoise continentalness_2B;
        ImprovedNoise continentalness_3A;
        ImprovedNoise continentalness_3B;
        ImprovedNoise continentalness_4A;
        ImprovedNoise continentalness_4B;
        ImprovedNoise continentalness_5A;
        ImprovedNoise continentalness_5B;
        ImprovedNoise continentalness_6A;
        ImprovedNoise continentalness_6B;
        ImprovedNoise continentalness_7A;
        ImprovedNoise continentalness_7B;
        ImprovedNoise continentalness_8A;
        ImprovedNoise continentalness_8B;
    };

    template<size_t Octaves>
    struct ResultSampler {
        ImprovedNoise octaves[Octaves];

        __device__ float sample(const GradDotTable &table, int32_t x, int32_t y, int32_t z) const {
            float val = 0;
            if constexpr (Octaves >=  1) val += sample_octave<chosen_continentalness_config.octaves_a[0]>(table, octaves[ 0], x, y, z);
            if constexpr (Octaves >=  2) val += sample_octave<chosen_continentalness_config.octaves_b[0]>(table, octaves[ 1], x, y, z);
            if constexpr (Octaves >=  3) val += sample_octave<chosen_continentalness_config.octaves_a[1]>(table, octaves[ 2], x, y, z);
            if constexpr (Octaves >=  4) val += sample_octave<chosen_continentalness_config.octaves_b[1]>(table, octaves[ 3], x, y, z);
            if constexpr (Octaves >=  5) val += sample_octave<chosen_continentalness_config.octaves_a[2]>(table, octaves[ 4], x, y, z);
            if constexpr (Octaves >=  6) val += sample_octave<chosen_continentalness_config.octaves_b[2]>(table, octaves[ 5], x, y, z);
            if constexpr (Octaves >=  7) val += sample_octave<chosen_continentalness_config.octaves_a[3]>(table, octaves[ 6], x, y, z);
            if constexpr (Octaves >=  8) val += sample_octave<chosen_continentalness_config.octaves_b[3]>(table, octaves[ 7], x, y, z);
            if constexpr (Octaves >=  9) val += sample_octave<chosen_continentalness_config.octaves_a[4]>(table, octaves[ 8], x, y, z);
            if constexpr (Octaves >= 10) val += sample_octave<chosen_continentalness_config.octaves_b[4]>(table, octaves[ 9], x, y, z);
            if constexpr (Octaves >= 11) val += sample_octave<chosen_continentalness_config.octaves_a[5]>(table, octaves[10], x, y, z);
            if constexpr (Octaves >= 12) val += sample_octave<chosen_continentalness_config.octaves_b[5]>(table, octaves[11], x, y, z);
            if constexpr (Octaves >= 13) val += sample_octave<chosen_continentalness_config.octaves_a[6]>(table, octaves[12], x, y, z);
            if constexpr (Octaves >= 14) val += sample_octave<chosen_continentalness_config.octaves_b[6]>(table, octaves[13], x, y, z);
            if constexpr (Octaves >= 15) val += sample_octave<chosen_continentalness_config.octaves_a[7]>(table, octaves[14], x, y, z);
            if constexpr (Octaves >= 16) val += sample_octave<chosen_continentalness_config.octaves_b[7]>(table, octaves[15], x, y, z);
            if constexpr (Octaves >= 17) val += sample_octave<chosen_continentalness_config.octaves_a[8]>(table, octaves[16], x, y, z);
            if constexpr (Octaves >= 18) val += sample_octave<chosen_continentalness_config.octaves_b[8]>(table, octaves[17], x, y, z);
            return val;
        }
    };

    __device__ Result results[threads_per_run];
    // constexpr size_t a = sizeof(results);

    __device__ void copy_noise(ImprovedNoise (&shared_noise)[threads_per_block], ImprovedNoise Result::* result_member) {
        for (uint32_t result_index = 0; result_index < threads_per_block; result_index++) {
            ImprovedNoise &src = shared_noise[result_index];
            ImprovedNoise &dst = results[blockIdx.x * blockDim.x + result_index].*result_member;
            for (uint32_t i = threadIdx.x; i < sizeof(ImprovedNoise) / sizeof(uint32_t); i += threads_per_block) {
                reinterpret_cast<uint32_t*>(&dst)[i] = reinterpret_cast<uint32_t*>(&src)[i];
            }
        }
    }

    __device__ void init_octave(const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, ImprovedNoise Result::* result_member) {
        __shared__ ImprovedNoise shared_noise[threads_per_block];

        init_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
        __syncthreads();
        copy_noise(shared_noise, result_member);
        __syncthreads();
    }

    __global__ __launch_bounds__(threads_per_block) void kernel(uint64_t start_seed) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t seed = start_seed + index;

        const auto seed_fork = XrsrRandom(seed).fork();
        auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

        const auto noise_a_fork = noise_random.fork();
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[0].fork_hash, &Result::continentalness_0A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[1].fork_hash, &Result::continentalness_1A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[2].fork_hash, &Result::continentalness_2A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[3].fork_hash, &Result::continentalness_3A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[4].fork_hash, &Result::continentalness_4A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[5].fork_hash, &Result::continentalness_5A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[6].fork_hash, &Result::continentalness_6A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[7].fork_hash, &Result::continentalness_7A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[8].fork_hash, &Result::continentalness_8A);

        const auto noise_b_fork = noise_random.fork();
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[0].fork_hash, &Result::continentalness_0B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[1].fork_hash, &Result::continentalness_1B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[2].fork_hash, &Result::continentalness_2B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[3].fork_hash, &Result::continentalness_3B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[4].fork_hash, &Result::continentalness_4B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[5].fork_hash, &Result::continentalness_5B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[6].fork_hash, &Result::continentalness_6B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[7].fork_hash, &Result::continentalness_7B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[8].fork_hash, &Result::continentalness_8B);
    }
}

struct DeviceBuffer {
    void *data;
    size_t size;

    DeviceBuffer(size_t size) : size(size) {
        TRY_HIP(hipMalloc(&data, size));
    }

    ~DeviceBuffer() {
        TRY_HIP(hipFree(data));
    }
};

template<typename T>
struct OutputBuffer {
    T *data;
    uint32_t &len;
    uint32_t max_len;

    OutputBuffer(T *data, uint32_t &len, uint32_t max_len) : data(data), len(len), max_len(max_len) {

    }

    OutputBuffer(const DeviceBuffer &buffer, uint32_t &len) : data((T*)buffer.data), len(len), max_len(buffer.size / sizeof(T)) {

    }

    OutputBuffer(const OutputBuffer<T> &other) : data(other.data), len(other.len), max_len(other.max_len) {

    }
};

template<typename T>
struct InputBuffer {
    const T *data;
    const uint32_t &len;

    InputBuffer(const T *data, const uint32_t &len) : data(data), len(len) {

    }

    InputBuffer(const OutputBuffer<T> &buffer) : data(buffer.data), len(buffer.len) {

    }

    InputBuffer(const InputBuffer<T> &other) : data(other.data), len(other.len) {

    }
};

struct SeedPos {
    uint32_t seed_index;
    int32_t x;
    int32_t z;
};

constexpr int32_t small_biomes_pos_div = large_biomes ? 1 : 4;

namespace KernelFilter1 {
    constexpr uint32_t threads_per_block = 256;
    constexpr uint32_t threads_per_seed_sqrt = UINT64_C(1) << 10;
    constexpr uint32_t threads_per_seed = threads_per_seed_sqrt * threads_per_seed_sqrt;
    constexpr int32_t pos_step = 58400 / small_biomes_pos_div / 4;
    constexpr int32_t pos_range = (int32_t)threads_per_seed_sqrt * pos_step;
    static_assert(pos_range <= 60'000'000 / 4);

    __global__ __launch_bounds__(threads_per_block) void kernel(uint32_t start_seed_index, OutputBuffer<SeedPos> outputs) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t seed_index = start_seed_index + index / threads_per_seed;
        uint32_t pos_index = index % threads_per_seed;

        __shared__ GradDotTable shared_grad_dot_table;
        if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
            reinterpret_cast<uint32_t*>(&shared_grad_dot_table)[threadIdx.x] = reinterpret_cast<uint32_t*>(&device_grad_dot_table)[threadIdx.x];
        }

        __shared__ KernelSeed1::ResultSampler<2> shared_octaves;
        if (threadIdx.x < sizeof(shared_octaves) / sizeof(uint32_t)) {
            reinterpret_cast<uint32_t*>(&shared_octaves)[threadIdx.x] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[seed_index])[threadIdx.x];
        }
        __syncthreads();

        uint32_t x_index = pos_index % threads_per_seed_sqrt;
        uint32_t z_index = pos_index / threads_per_seed_sqrt;

        int32_t x = (int32_t)x_index * pos_step - pos_range / 2;
        int32_t z = (int32_t)z_index * pos_step - pos_range / 2;

        float val = shared_octaves.sample(shared_grad_dot_table, x, 0, z);

        if (val >= -0.515f) return; // 1 in 27.7
        // if (val >= -0.7f) return; // 1 in 176
        // if (val >= -0.8f) return;
        // if (val >= -1.48f) return;

        uint32_t result_index = atomicAdd(&outputs.len, 1);
        if (result_index >= outputs.max_len) return;
        outputs.data[result_index] = { seed_index, x, z };
    }

    void run(uint32_t start_seed_index, uint32_t seeds, OutputBuffer<SeedPos> outputs) {
        hipLaunchKernelGGL(kernel, dim3(seeds * threads_per_seed / threads_per_block), dim3(threads_per_block), 0, 0, start_seed_index, outputs);
        TRY_HIP(hipGetLastError());
    }
}

constexpr bool is_pow2(uint32_t val) {
    return (val & (val - 1)) == 0;
}

constexpr uint32_t log2(uint32_t val) {
    return 31 - std::countl_zero(val);
}

template<typename T>
__device__ T warp_reduce_add(T val) {
#if __HIP_DEVICE_COMPILE__ >= 800
    return __reduce_add_sync(0xFFFFFFFF, val);
#else
    val += __shfl_down(val, 1);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 16);
    return val;
#endif
}

namespace KernelFilter2 {
    template<int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter>
    struct Template {
        static constexpr float noise_threshold = NoiseThreshold / 10000.0f;
        static constexpr size_t octaves = Octaves;
        static constexpr uint32_t pos_range = PosRange;
        static constexpr uint32_t samples = Samples;
        static constexpr uint32_t min_count = MinCount;
        static constexpr bool flipped_sparse_samples = FlippedSparseSamples;
        static constexpr bool move_center = MoveCenter;

        static constexpr uint32_t threads_per_block = 256;
        static_assert(samples >= 32 && samples <= threads_per_block * threads_per_block && is_pow2(samples));
        static constexpr uint32_t samples_square_size = UINT32_C(1) << (log2(samples) + 1) / 2;
        static constexpr bool samples_square_sparse = log2(samples) % 2 == 1;
        static_assert(!flipped_sparse_samples || samples_square_sparse);
        static_assert(pos_range % (small_biomes_pos_div * 4 * samples_square_size * 2) == 0);
        static constexpr uint32_t pos_step = pos_range / small_biomes_pos_div / 4 / samples_square_size;
        static constexpr int32_t pos_offset = -(int32_t)(pos_step * (samples_square_size - 1) / 2);

        static constexpr uint32_t threads_per_input = std::min(samples, threads_per_block);
        static constexpr uint32_t loops = samples / threads_per_input;
        static constexpr uint32_t inputs_per_block = threads_per_block / threads_per_input;

        static void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs);
    };

    template<typename T>
    __global__ __launch_bounds__(T::threads_per_block) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        __shared__ GradDotTable shared_grad_dot_table;
        if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
            reinterpret_cast<uint32_t*>(&shared_grad_dot_table)[threadIdx.x] = reinterpret_cast<uint32_t*>(&device_grad_dot_table)[threadIdx.x];
        }

        uint32_t inputs_len = inputs.len;
        for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < inputs_len * T::threads_per_input; index += gridDim.x * blockDim.x) {
            __syncthreads();
            uint32_t input_index = index / T::threads_per_input;
            uint32_t pos_index = index % T::threads_per_input;
            uint32_t block_input_index = threadIdx.x / T::threads_per_input;

            const auto input = inputs.data[input_index];

            __shared__ KernelSeed1::ResultSampler<T::octaves> shared_octaves[T::inputs_per_block];
            for (uint32_t i = pos_index; i < sizeof(shared_octaves[0]) / sizeof(uint32_t); i += T::threads_per_input) {
                reinterpret_cast<uint32_t*>(&shared_octaves[block_input_index])[i] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[input.seed_index])[i];
            }
            __syncthreads();

            uint32_t x_index = pos_index % T::samples_square_size;
            uint32_t z_index = pos_index / T::samples_square_size;
            if constexpr (T::samples_square_sparse) {
                z_index = z_index * 2 + ((x_index & 1) ^ T::flipped_sparse_samples);
            }

            int32_t x = input.x + (int32_t)(x_index * T::pos_step) + T::pos_offset;
            int32_t z = input.z + (int32_t)(z_index * T::pos_step) + T::pos_offset;

            uint32_t total_valid = 0;
            int32_t sum_dx = 0;
            int32_t sum_dz = 0;

            for (uint32_t i = 0; i < T::loops; i++) {
                float val = shared_octaves[block_input_index].sample(shared_grad_dot_table, x, 0, z);

                bool valid = val < T::noise_threshold;

                total_valid += warp_reduce_add((uint32_t)valid);

                if constexpr (T::move_center) {
                    if (valid) {
                        sum_dx += x - input.x;
                        sum_dz += z - input.z;
                    }
                }

                z += (int32_t)(T::pos_step * (T::samples_square_size / T::loops));
            }

            if constexpr (T::samples > 32) {
                __shared__ uint32_t shared_counts[T::inputs_per_block];
                if (threadIdx.x < T::inputs_per_block) {
                    shared_counts[threadIdx.x] = 0;
                }
                __syncthreads();
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&shared_counts[block_input_index], total_valid);
                }
                __syncthreads();
                total_valid = shared_counts[block_input_index];
            }

            if constexpr (T::move_center) {
                sum_dx = warp_reduce_add(sum_dx);
                sum_dz = warp_reduce_add(sum_dz);
                if constexpr (T::samples > 32) {
                    __shared__ int32_t shared_sums[T::inputs_per_block][2];
                    if (threadIdx.x < T::inputs_per_block) {
                        shared_sums[threadIdx.x][0] = 0;
                        shared_sums[threadIdx.x][1] = 0;
                    }
                    __syncthreads();
                    if (threadIdx.x % 32 == 0) {
                        atomicAdd(&shared_sums[block_input_index][0], sum_dx);
                        atomicAdd(&shared_sums[block_input_index][1], sum_dz);
                    }
                    __syncthreads();
                    sum_dx = shared_sums[block_input_index][0];
                    sum_dz = shared_sums[block_input_index][1];
                }
                if (total_valid != 0) {
                    sum_dx /= (int32_t)total_valid;
                    sum_dz /= (int32_t)total_valid;
                }
            }

            if (total_valid < T::min_count) continue;

            if (pos_index == 0) {
                uint32_t result_index = atomicAdd(&outputs.len, 1);
                if (result_index >= outputs.max_len) continue;
                outputs.data[result_index] = { input.seed_index, input.x + sum_dx, input.z + sum_dz };
            }
        }
    }

    template<int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter>
    void Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter>::run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        using T = Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter>;
        hipLaunchKernelGGL(kernel<T>, dim3(32 * 256), dim3(T::threads_per_block), 0, 0, inputs, outputs);
        TRY_HIP(hipGetLastError());
    }
}

struct HipEventWrapper {
    hipEvent_t event;

    HipEventWrapper() : event(nullptr) {
        TRY_HIP(hipEventCreate(&event));
    }

    HipEventWrapper(HipEventWrapper &&other) : event(other.event) {
        other.event = nullptr;
    }

    ~HipEventWrapper() {
        if (event == nullptr) return;
        TRY_HIP(hipEventDestroy(event));
    }

    void record(hipStream_t stream = 0) const {
        TRY_HIP(hipEventRecord(event, stream));
    }

    float elapsed(const HipEventWrapper &end) const {
        float ms;
        TRY_HIP(hipEventElapsedTime(&ms, event, end.event));
        return ms;
    }

    void synchronize() const {
        TRY_HIP(hipEventSynchronize(event));
    }
};

struct BufferLens {
    uint32_t results_len_filter_1;
    uint32_t results_len_filter_2[7];
};

uint64_t random_start_seed() {
    std::random_device device;
    return ((uint64_t)device() << 32) + (uint64_t)device();
}

GpuThread::GpuThread(int device, GpuOutputs &outputs) : Thread(), device(device), outputs(outputs) {
    start();
}

void GpuThread::run() {
    std::printf("Initializing device %d\n", device);

    TRY_HIP(hipSetDevice(device));
    init_grad_dot_table();

    BufferLens host_buffer_lens;
    BufferLens *device_buffer_lens;
    TRY_HIP(hipMalloc(&device_buffer_lens, sizeof(*device_buffer_lens)));

    uint64_t start_seed = random_start_seed();
    // uint64_t start_seed = 9849470875906027758; i'm leaving this here for myself as well but if you want to set a set seed
	// after you've stopped then just remove the random_start_seed() ^ and replace it with a number
    std::printf("Running device %d at %" PRIu64 "\n", device, start_seed);

    DeviceBuffer buffer_1(UINT32_C(1) << 31);
    DeviceBuffer buffer_2(UINT32_C(1) << 29);
    std::vector<SeedPos> h_buffer;

    namespace Filter1 = KernelFilter1;

    using Kernel2RunFunc = void (*)(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs);

    struct Filter2Stage {
        char letter;
        Kernel2RunFunc run;
        OutputBuffer<SeedPos> outputs;
        uint32_t &host_outputs_len;
        HipEventWrapper event;
        double time;
        uint64_t total_outputs_len;

        Filter2Stage(char letter, Kernel2RunFunc run, OutputBuffer<SeedPos> outputs, uint32_t &host_outputs_len, HipEventWrapper &&event, double time, uint64_t total_outputs_len) : letter(letter), run(run), outputs(outputs), host_outputs_len(host_outputs_len), event(std::move(event)), time(time), total_outputs_len(total_outputs_len) {

        }
    };
    // Kernel2RunFunc filter_2_runs[] = {
    //     KernelFilter2::Template<-7400, 2, 24 * 1024, 32, 2, false, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = 0.0624
    //     KernelFilter2::Template<-7400, 2, 24 * 1024, 128, 10, true, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = 0.0197
    //     KernelFilter2::Template<-7400, 2, 24 * 1024, 512, 51, false, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = <0.01
    //     KernelFilter2::Template<-10500, 18, 30 * 1024, 256, 14, false, true>::run, // 88m -> 0.09324815538194445 | P(X < x) = <0.01
    //     KernelFilter2::Template<-10500, 18, 30 * 1024, 8192, 703, false, false>::run, // 88m -> 0.09324815538194445 | P(X < x) = <0.01
    // };
    Kernel2RunFunc filter_2_runs[] = {
        KernelFilter2::Template<-7400, 2, 24 * 1024, 32, 2, false, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = 0.0624
        KernelFilter2::Template<-7400, 2, 24 * 1024, 128, 10, true, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = 0.0197
        KernelFilter2::Template<-7400, 2, 24 * 1024, 512, 51, false, true>::run, // 80m -> 0.1324547661675347 | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 32 * 1024, 256, 11, false, true>::run, // 88m -> 0.08195638656616211 | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 32 * 1024, 8192, 614, false, false>::run, // 88m -> 0.08195638656616211 | P(X < x) = <0.01
    };
    size_t filter_2_runs_len = sizeof(filter_2_runs) / sizeof(*filter_2_runs);
    std::vector<Filter2Stage> filter_2;
    filter_2.reserve(filter_2_runs_len);
    for (size_t i = 0; i < filter_2_runs_len; i++) {
        filter_2.emplace_back((char)('a' + i), filter_2_runs[i], OutputBuffer<SeedPos>(i % 2 == 0 ? buffer_2 : buffer_1, device_buffer_lens->results_len_filter_2[i]), host_buffer_lens.results_len_filter_2[i], HipEventWrapper(), 0.0, 0);
    }

    OutputBuffer<SeedPos> outputs_filter_1(buffer_1, device_buffer_lens->results_len_filter_1);
    uint32_t &host_outputs_filter_1_len = host_buffer_lens.results_len_filter_1;

    HipEventWrapper event_start, event_seed_1, event_filter_1;

    int print_interval = 64;
    double time_seed_1 = 0.0;
    double time_filter_1 = 0.0;
    uint64_t inputs_seed_1 = 0;
    uint64_t inputs_filter_1 = 0;
    uint64_t total_outputs_len_filter_1 = 0;

    auto start = std::chrono::steady_clock::now();

    uint64_t currently_used_start_seed = 0;
    uint32_t start_seed_index = UINT32_MAX;

    for (uint32_t i = 0; !should_stop(); i++) {
        TRY_HIP(hipMemsetAsync(device_buffer_lens, 0, sizeof(*device_buffer_lens), 0));

        event_start.record();

        if (start_seed_index >= KernelSeed1::threads_per_run) {
            inputs_seed_1 += KernelSeed1::threads_per_run;
            hipLaunchKernelGGL(KernelSeed1::kernel, dim3(KernelSeed1::threads_per_run / KernelSeed1::threads_per_block), dim3(KernelSeed1::threads_per_block), 0, 0, start_seed);
            TRY_HIP(hipGetLastError());
            currently_used_start_seed = start_seed;
            start_seed += KernelSeed1::threads_per_run;
            start_seed_index = 0;
        }
        event_seed_1.record();

        {
            uint32_t seeds = std::min(KernelSeed1::threads_per_run - start_seed_index, outputs_filter_1.max_len * 16 / Filter1::threads_per_seed);
            inputs_filter_1 += seeds * Filter1::threads_per_seed;
            Filter1::run(start_seed_index, seeds, outputs_filter_1);
            start_seed_index += seeds;
        }
        event_filter_1.record();

        {
            OutputBuffer<SeedPos> *inputs = &outputs_filter_1;
            for (auto &stage : filter_2) {
                stage.run(*inputs, stage.outputs);
                stage.event.record();
                inputs = &stage.outputs;
            }
        }

        TRY_HIP(hipMemcpyAsync(&host_buffer_lens, device_buffer_lens, sizeof(host_buffer_lens), hipMemcpyDeviceToHost));

        filter_2.back().event.synchronize();

        time_seed_1 += event_start.elapsed(event_seed_1) * 1e-3;
        time_filter_1 += event_seed_1.elapsed(event_filter_1) * 1e-3;
        total_outputs_len_filter_1 += host_outputs_filter_1_len;
        {
            HipEventWrapper *start_event = &event_filter_1;
            for (auto &stage : filter_2) {
                stage.time += start_event->elapsed(stage.event) * 1e-3;
                start_event = &stage.event;

                stage.total_outputs_len += stage.host_outputs_len;
            }
        }

        if (host_outputs_filter_1_len > outputs_filter_1.max_len) {
            std::printf("outputs_filter_1.len > outputs_filter_1.max_len : %" PRIu32 " > %" PRIu32 "\n", host_outputs_filter_1_len, outputs_filter_1.max_len);
        }
        for (auto &stage : filter_2) {
            if (stage.host_outputs_len > stage.outputs.max_len) {
                std::printf("outputs_filter_2%c.len > outputs_filter_2%c.max_len : %" PRIu32 " > %" PRIu32 "\n", stage.letter, stage.letter, stage.host_outputs_len, stage.outputs.max_len);
            }
        }

        const auto &final_outputs = filter_2.back().outputs;
        const auto &final_outputs_len = filter_2.back().host_outputs_len;
        if (final_outputs_len > 0) {
            // uint32_t len = std::min(final_outputs_len, UINT32_C(10));
            uint32_t len = final_outputs_len;
            h_buffer.resize(len);
            TRY_HIP(hipMemcpy(h_buffer.data(), final_outputs.data, sizeof(*h_buffer.data()) * len, hipMemcpyDeviceToHost));

            {
                std::lock_guard lock(outputs.mutex);
                for (const auto &result : h_buffer) {
                    outputs.queue.push({ currently_used_start_seed + result.seed_index, result.x * 4, result.z * 4 });
                }
            }
        }

        if ((i + 1) % print_interval == 0) {
            auto end = std::chrono::steady_clock::now();
            double time_total = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

            double kernel_time_total = time_seed_1 + time_filter_1;
            std::printf("\n");
            std::printf("seed_1    - %9.3f ms | %6.3f %% | %12" PRIu64 "                "" |                 "" | %7.3f Msps\n", time_seed_1 * 1e3, time_seed_1 / time_total * 100.0, inputs_seed_1, inputs_seed_1 / time_seed_1 * 1e-6);
            std::printf("filter_1  - %9.3f ms | %6.3f %% | %12" PRIu64 " -> %12" PRIu64  " | 1 in %11.3f"     " | %7.3f Gsps\n", time_filter_1 * 1e3, time_filter_1 / time_total * 100.0, inputs_filter_1, total_outputs_len_filter_1, (double)inputs_filter_1 / total_outputs_len_filter_1, inputs_filter_1 / time_filter_1 * 1e-9);
            uint64_t inputs = total_outputs_len_filter_1;
            for (auto &stage : filter_2) {
                std::printf("filter_2%c - %9.3f ms | %6.3f %% | %12" PRIu64 " -> %12" PRIu64  " | 1 in %11.3f"     " | %7.3f Msps\n", stage.letter, stage.time * 1e3, stage.time / time_total * 100.0, inputs, stage.total_outputs_len, (double)inputs / stage.total_outputs_len, inputs / stage.time * 1e-6);
                kernel_time_total += stage.time;
                inputs = stage.total_outputs_len;
            }
            std::printf("total     - %9.3f ms | %6.3f %% |                             " " |                 "" | %7.3f Gsps | %6.3f Msps\n", time_total * 1e3, kernel_time_total / time_total * 100.0, inputs_filter_1 / time_total * 1e-9, inputs_filter_1 / time_total / 81.0 * 1e-6);
            size_t gpu_outputs_size;
            {
                std::lock_guard lock(outputs.mutex);
                gpu_outputs_size = outputs.queue.size();
            }
            std::printf("gpu_outputs.size() = %zu\n", gpu_outputs_size);

            start = end;
            time_seed_1 = 0.0;
            time_filter_1 = 0.0;
            inputs_seed_1 = 0;
            inputs_filter_1 = 0;
            total_outputs_len_filter_1 = 0;
            for (auto &stage : filter_2) {
                stage.time = 0;
                stage.total_outputs_len = 0;
            }
        }
    }

    TRY_HIP(hipFree(device_buffer_lens));
}