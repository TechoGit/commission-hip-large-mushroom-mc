#pragma once

#include "common.h"

struct CpuThread: Thread<CpuThread> {
    int id;
    GpuOutputs &inputs;
    CpuOutputs &outputs;

    CpuThread(int id, GpuOutputs &inputs, CpuOutputs &outputs);

    void run();
};