#include "cuadmm/kernels.h"

TEST(DenseScalar, Simple)
{
    DeviceStream stream;
    stream.set_gpu_id(0);
    stream.activate();

    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate(stream);

    DeviceDenseVector<double> dvec;
    dvec.allocate(0, 10);
}

TEST(DenseScalar, Norm)
{

}