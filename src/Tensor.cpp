#include "../include/Tensor.hpp"

void Tensor::allocate_memory() {
    if (m_size == 0) return;

    size_t bytes = m_size * sizeof(float);

#if defined(_MSC_VER) || defined(__MINGW32__)
    m_data = static_cast<float*>(_aligned_malloc(bytes, ALIGNMENT_BYTES));
    if (!m_data) throw std::bad_alloc();
#else
    if (posix_memalign(reinterpret_cast<void**>(&m_data), ALIGNMENT_BYTES, bytes) != 0) {
        throw std::bad_alloc();
    }
#endif
}

void Tensor::compute_strides() {
    m_strides.resize(m_shape.size());
    if (m_shape.empty()) return;

    size_t stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

Tensor::Tensor(std::vector<size_t> shape) : m_shape(std::move(shape)) {
    m_size = 1;
    for (auto dim : m_shape) m_size *= dim;
    compute_strides();
    allocate_memory();
    std::memset(m_data, 0, m_size * sizeof(float));
}

Tensor::Tensor(std::vector<size_t> shape, float fill_value) : m_shape(std::move(shape)) {
    m_size = 1;
    for (auto dim : m_shape) m_size *= dim;
    compute_strides();
    allocate_memory();
    for (size_t i = 0; i < m_size; ++i) m_data[i] = fill_value;
}

Tensor::Tensor(std::initializer_list<size_t> shape) : Tensor(std::vector<size_t>(shape)) {}

Tensor::~Tensor() {
    if (m_data) {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(m_data);
#else
        free(m_data);
#endif
        m_data = nullptr;
    }
}

Tensor::Tensor(const Tensor& other) : m_shape(other.m_shape), m_strides(other.m_strides), m_size(other.m_size) {
    allocate_memory();
    std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (m_data) {
#if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(m_data);
#else
            free(m_data);
#endif
        }
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_size = other.m_size;
        allocate_memory();
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : m_data(other.m_data), m_size(other.m_size), m_shape(std::move(other.m_shape)), m_strides(std::move(other.m_strides)) {
    other.m_data = nullptr;
    other.m_size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (m_data) {
#if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(m_data);
#else
            free(m_data);
#endif
        }
        m_data = other.m_data;
        m_size = other.m_size;
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);

        other.m_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < m_shape.size(); ++i) {
        std::cout << m_shape[i] << (i + 1 < m_shape.size() ? ", " : "");
    }
    std::cout << "], Size: " << m_size << "\n";
}