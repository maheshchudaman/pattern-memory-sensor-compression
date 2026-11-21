#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sprintz_delta.h"

template <typename T>
bool read_file(const std::string& path, std::vector<T>* values) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) return false;
  const auto bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) return false;
  values->resize(static_cast<size_t>(bytes) / sizeof(T));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(values->data()), bytes);
  return static_cast<bool>(input);
}

template <typename T>
bool write_file(const std::string& path, const T* values, size_t count) {
  std::ofstream output(path, std::ios::binary);
  if (!output) return false;
  output.write(reinterpret_cast<const char*>(values), count * sizeof(T));
  return static_cast<bool>(output);
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: sprintz_cli <compress|decompress> <input> <output> "
                 "<ndims> <uncompressed_elements>\n";
    return 2;
  }
  const std::string mode = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];
  const uint16_t ndims = static_cast<uint16_t>(std::stoul(argv[4]));
  const uint32_t elements = static_cast<uint32_t>(std::stoul(argv[5]));

  if (mode == "compress") {
    std::vector<uint16_t> input;
    if (!read_file(input_path, &input) || input.size() != elements) return 3;
    // The official implementation may briefly write a vector past the logical
    // end, so retain generous padding and trim to the returned element count.
    std::vector<int16_t> output(input.size() * 2 + 4096);
    const int64_t count = compress_rowmajor_delta_rle_16b(
        input.data(), elements, output.data(), ndims, true);
    if (count < 0 || !write_file(output_path, output.data(), count)) return 4;
    std::cout << count * sizeof(int16_t) << "\n";
    return 0;
  }
  if (mode == "decompress") {
    std::vector<int16_t> input;
    if (!read_file(input_path, &input)) return 5;
    std::vector<uint16_t> output(static_cast<size_t>(elements) + 4096);
    const int64_t count = decompress_rowmajor_delta_rle_16b(input.data(), output.data());
    if (count != elements || !write_file(output_path, output.data(), elements)) return 6;
    std::cout << count << "\n";
    return 0;
  }
  return 7;
}
