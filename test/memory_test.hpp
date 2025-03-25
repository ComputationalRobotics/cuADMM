#include "cuadmm/memory.h"

TEST(Memory, DeviceStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();
    device_stream.~DeviceStream();
}

TEST(Memory, SimpleDeviceBlasHandle)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate();
    handle.~DeviceBlasHandle();
}

TEST(Memory, DeviceBlasHandleWithStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();

    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate(device_stream);
    handle.~DeviceBlasHandle();
    device_stream.~DeviceStream();
}

TEST(Memory, DeviceSolverDnHandle)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(0);
    handle.activate();
    handle.~DeviceSolverDnHandle();
}

TEST(Memory, DeviceSolverDnHandleWithStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();

    DeviceSolverDnHandle handle;
    handle.set_gpu_id(0);
    handle.activate(device_stream);
    handle.~DeviceSolverDnHandle();
    device_stream.~DeviceStream();
}