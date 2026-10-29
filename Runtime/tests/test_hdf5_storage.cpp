// Manual native I/O regression; link against Hdf5File.o and Logger.o plus HDF5.
#include <Hdf5/Hdf5File.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA_STORAGE
#include <MatrixClasses/ComplexMatrix.h>
#endif

int main(int argc, char** argv) {
    if (argc != 2) throw std::invalid_argument("usage: test_hdf5_storage output.h5");
    Hdf5File file;
    file.create(argv[1]);
    const hid_t root = file.getRootGroup();
    const std::string longValue(1024, 'x');
    file.writeStringAttribute(root, "/", "long_string", longValue);
    assert(file.readStringAttribute(root, "/", "long_string") == longValue);
    for (int rank : {3, 4, 5}) {
        std::vector<hsize_t> dims(rank, 2);
        hid_t space = H5Screate_simple(rank, dims.data(), nullptr);
        std::string name = "rank" + std::to_string(rank);
        hid_t data = H5Dcreate2(root, name.c_str(), H5T_NATIVE_FLOAT, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(data);
        H5Sclose(space);
        assert(file.getDatasetSize(root, name) == (size_t(1) << rank));
        if (rank <= 4) {
            auto shape = file.getDatasetDimensionSizes(root, name);
            assert(shape.nx == 2 && shape.ny == 2 && shape.nz == 2);
        } else {
            bool rejected = false;
            try { file.getDatasetDimensionSizes(root, name); }
            catch (const std::ios::failure&) { rejected = true; }
            assert(rejected);
        }
    }
    // Scalar writes and failed hyperslabs must not retain open HDF5 handles.
    const auto openObjects = H5Fget_obj_count(root, H5F_OBJ_ALL);
    for (int i = 0; i < 5; ++i) file.writeScalarValue<float>(root, "scalar", float(i));
    assert(H5Fget_obj_count(root, H5F_OBJ_ALL) == openObjects);
    hid_t rank5 = file.openDataset(root, "rank5");
    const auto beforeFailure = H5Fget_obj_count(root, H5F_OBJ_ALL);
    float unused = 0;
    bool rejectedSlab = false;
    try { file.writeHyperSlab(rank5, DimensionSizes(0,0,0), DimensionSizes(1,1,1), &unused); }
    catch (const std::ios::failure&) { rejectedSlab = true; }
    assert(rejectedSlab);
    assert(H5Fget_obj_count(root, H5F_OBJ_ALL) == beforeFailure);
    file.closeDataset(rank5);
    auto dataset = file.createDataset(root, "values", DimensionSizes(6, 3, 2),
                                      DimensionSizes(6, 3, 1), Hdf5File::MatrixDataType::kFloat, 0);
    std::vector<float> values(36, 7);
    file.writeHyperSlab(dataset, DimensionSizes(0, 0, 0), DimensionSizes(6, 3, 2), values.data());
    file.closeDataset(dataset);
    std::vector<float> restored(36);
    file.readCompleteDataset(root, "values", DimensionSizes(6, 3, 2), restored.data());
    assert(values == restored);
#ifdef TEST_CUDA_STORAGE
    ComplexMatrix complexValues(DimensionSizes(3, 2, 2));
    for (size_t i = 0; i < complexValues.capacity(); ++i) complexValues.getHostData()[i] = float(i);
    complexValues.writeData(file, "complex_values", 0);
    const auto complexShape = file.getDatasetDimensionSizes(root, "complex_values");
    assert(complexShape.nx == 6 && complexShape.ny == 2 && complexShape.nz == 2);
    std::vector<float> complexRestored(complexValues.capacity());
    file.readCompleteDataset(root, "complex_values", complexShape, complexRestored.data());
    for (size_t i = 0; i < complexRestored.size(); ++i) assert(complexRestored[i] == float(i));
#endif
    file.close();
    file.close();
    std::cout << "HDF5 rank, long attribute, slab and close checks passed\n";
}
