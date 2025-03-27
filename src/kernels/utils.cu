#include <iostream>

#include "cuadmm/utils.h"
#include "cuadmm/check.h"

// Check GPU numbers and whether they have peer-to-peer access
int check_gpus() {
    // count the number of GPUs
    int device_count = 0;
    CHECK_CUDA( cudaGetDeviceCount(&device_count) );
    std::cout << "Detected " << device_count << " CUDA Capable device(s)" << std::endl;

    // print the name of each GPU
    for (int dev = 0; dev < device_count; dev++) {
        CHECK_CUDA( cudaSetDevice(dev) );
        cudaDeviceProp deviceProp;
        CHECK_CUDA( cudaGetDeviceProperties_v2(&deviceProp, dev) );
        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
    }

    // set gpu 0 as the master, gpu 1 ... gpu (device_count-1) as slaves
    int can_access_peer;
    for (int i = 1; i < device_count; i++) {
        CHECK_CUDA( cudaDeviceCanAccessPeer(&can_access_peer, GPU0, i) );
        if (can_access_peer) {
            std::cout << "Device 0 can access Device " << i << " directly." << std::endl;
            CHECK_CUDA( cudaSetDevice(GPU0) );
            CHECK_CUDA( cudaDeviceEnablePeerAccess(i, 0) );
            CHECK_CUDA( cudaSetDevice(i) );
            CHECK_CUDA( cudaDeviceEnablePeerAccess(GPU0, 0) );
        } else {
            std::cerr << "Device 0 cannot access Device " << i << " directly." << std::endl;
            std::abort();
        }
    }
    return device_count;
}