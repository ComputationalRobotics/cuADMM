TEST(SparseDense, DivNorm)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> sp_vals(SIZE, 1.0);
    std::vector<int> indices;
    for (int i = 0; i < SIZE; i++) {
        indices.push_back(2*i);
    }
    
    // create the sparse vector
    DeviceSparseVector<double> sparse_vector(GPU0, 2*SIZE, SIZE);
    CHECK_CUDA( cudaMemcpy(sparse_vector.vals, sp_vals.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(sparse_vector.indices, indices.data(), sizeof(int) * SIZE, cudaMemcpyHostToDevice) );

    // create the dense vector
    std::vector<double> dense_vals(2*SIZE, 2.0);
    DeviceDenseVector<double> dense_vector(GPU0, 2*SIZE);
    CHECK_CUDA( cudaMemcpy(dense_vector.vals, dense_vals.data(), sizeof(double) * 2 * SIZE, cudaMemcpyHostToDevice) );

    // divide the vector by 2
    sparse_vector_div_dense_vector(sparse_vector, dense_vector);

    sparse_vector.print(true);

    //check the norm
    EXPECT_DOUBLE_EQ(sparse_vector.get_norm(handle), sqrt(SIZE/(2.0*2.0)));
}

TEST(SparseDense, DivSeries)
{

}