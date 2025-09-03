#pragma once

#include "common.h"

struct GpuThread: Thread<GpuThread> {
    int device;
    GpuOutputs &outputs;

    GpuThread(int device, GpuOutputs &outputs);

    void run();
};