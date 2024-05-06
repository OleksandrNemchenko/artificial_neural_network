
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_INTERNALS_
#define _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_INTERNALS_

#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <CL/opencl.hpp>

#include <halfFloat.hpp>
#include <annMetaConfigurations.hpp>

#include <artificial_neural_network/utilities.hpp>

using namespace std::string_literals;

#ifndef ANN_GPU_CALCULATIONS

namespace artificial_neural_network::openClEmulator
{

extern std::mutex global_objects_mutex;     // mutex for calculations
extern std::vector<size_t> global_id;       // global index
extern std::vector<size_t> global_size;     // global range
extern std::vector<size_t> local_id;        // local index within group
extern std::vector<size_t> local_size;      // group size
extern std::vector<size_t> num_groups;      // number of groups
extern std::vector<size_t> group_id;        // group ID
extern std::vector<size_t> global_offset;   // global offset
//extern size_t get_work_dim;                 // number of dimensions in use

};  // namespace artificial_neural_network::openClEmulator

#endif // ANN_GPU_CALCULATIONS

namespace artificial_neural_network
{

    static constexpr bool isDebug =
#ifdef _DEBUG
        true;
#else // _DEBUG
        false;
#endif // _DEBUG

// OPENCL CODE BEGINNING

typedef struct
{
    activation_function_type _neuronType;   // TODO: move to the layer settings
//    offset_type _layerNeuronPosition;     // TODO: for softmax
    offset_type _inputsAmount;
    offset_type _firstInputOff;
    offset_type _firstConfigOff;
    offset_type _stateOff;
} SNeuron;

inline offset_type Index1of2(const offset_type index, const offset_type size1) { return index % size1; }
inline offset_type Index2of2(const offset_type index, const offset_type size1) { return index / size1; }

inline offset_type Index1of3(const offset_type index, const offset_type size1) { return index % size1; }
inline offset_type Index2of3(const offset_type index, const offset_type size1, const offset_type size2) { return (index / size1) % size2; }
inline offset_type Index3of3(const offset_type index, const offset_type size1, const offset_type size2) { return index / size1 / size2; }

inline offset_type Index1of4(const offset_type index, const offset_type size1) { return index % size1; }
inline offset_type Index2of4(const offset_type index, const offset_type size1, const offset_type size2) { return (index / size1) % size2; }
inline offset_type Index3of4(const offset_type index, const offset_type size1, const offset_type size2, const offset_type size3) { return (index / size1 / size2) % size3; }
inline offset_type Index4of4(const offset_type index, const offset_type size1, const offset_type size2, const offset_type size3) { return index / size1 / size2 / size3; }

#ifdef ANN_OPENCL_CODE
#define assert(ACT) while(false)
#endif // ANN_OPENCL_CODE

// OPENCL CODE ENDING

#ifdef ANN_GPU_CALCULATIONS
    #define GPU_Action(x)   do{ x; } while(0)
    #define CPU_Action(x)   while(0)
#else // ANN_GPU_CALCULATIONS
    #define GPU_Action(x)   while(0)
    #define CPU_Action(x)   do{ x; } while(0)
#endif // ANN_GPU_CALCULATIONS

template<typename TSrc, typename TDst>
inline TDst Convert([[maybe_unused]] const TSrc value, [[maybe_unused]] long double maxDifference = 0)
{
#if ((defined(DEBUG) || defined(_DEBUG)) && !defined(ANN_CONVERT_ERROR_ASSERT)) || (!defined(DEBUG) && !defined(_DEBUG) && !defined(ANN_CONVERT_ERROR_THROW))

    return static_cast<TDst>(value);

#else // ((defined(DEBUG) || defined(_DEBUG)) && !defined(ANN_CONVERT_ERROR_ASSERT)) || (!defined(DEBUG) && !defined(_DEBUG) && !defined(ANN_CONVERT_ERROR_THROW))

    using namespace std::string_literals;

    if constexpr (std::is_floating_point_v<TSrc>)
    {
        const TSrc src = value;
#pragma warning(disable: 4244)
        const TDst dst = static_cast<TDst>(src);
#pragma warning(default: 4244)
        const long double diff = std::abs(static_cast<long double>(src) - dst);
        const long double maxValue = std::max(static_cast<long double>(src), static_cast<long double>(dst));

        if (!maxValue)
            return dst;

        const long double relativeDiff = diff / maxValue;

        if (maxDifference != 0 && relativeDiff > maxDifference)
        {
#ifdef ANN_CONVERT_ERROR_ASSERT
            assert(false);
#endif // ANN_CONVERT_ERROR_ASSERT

#ifdef ANN_CONVERT_ERROR_THROW
            throw std::runtime_error("Src data value "s + std::to_string(src) +
                " after converting to the required type is equal to "s + std::to_string(dst) +
                " and the difference "s + std::to_string(diff) + " (relative one is "s + std::to_string(relativeDiff) + ") "s +
                " is bigger that the required "s + std::to_string(maxDifference) + " one"s);
#endif // ANN_CONVERT_ERROR_THROW
        }

        return dst;
    }

    else if constexpr (std::is_same_v<std::remove_all_extents_t<TDst>, ext_data_array>)
    {
        ext_data_array resArray;
        resArray.reserve(value.size());

        for (const auto res : value)
            resArray.emplace_back(res);

        return resArray;
    }

    else if constexpr (std::is_integral_v<TSrc>)
    {
        if (value > std::numeric_limits<TDst>::max())
        {
#ifdef ANN_CONVERT_ERROR_ASSERT
            assert(false);
#endif // ANN_CONVERT_ERROR_ASSERT

#ifdef ANN_CONVERT_ERROR_THROW
            throw std::runtime_error("Src data value "s + std::to_string(value) + " is bigger than dst maximum value "s + std::to_string(std::numeric_limits<TDst>::max()));
#endif // ANN_CONVERT_ERROR_THROW
        }

        return static_cast<TDst>(value);
    }

    else
        return TDst();
#endif // ((defined(DEBUG) || defined(_DEBUG)) && !defined(ANN_CONVERT_ERROR_ASSERT)) || (!defined(DEBUG) && !defined(_DEBUG) && !defined(ANN_CONVERT_ERROR_THROW))
}

inline void WaitEvent([[maybe_unused]] cl::Event& event)
{
#ifdef ANN_GPU_CALCULATIONS
    event.wait();
#endif // ANN_GPU_CALCULATIONS
}

inline void CheckClError([[maybe_unused]] cl_int clError, [[maybe_unused]] const std::string& errDescr)
{
#ifdef ANN_GPU_CALCULATIONS
    if (clError != CL_SUCCESS)
        throw std::runtime_error(errDescr + ". Error code "s + std::to_string(clError));
#endif // ANN_GPU_CALCULATIONS
}

//#ifdef ANN_GPU_CALCULATIONS
cl::Device FindDevice(std::string_view deviceName);
//#endif // ANN_GPU_CALCULATIONS

template <typename TData>
class CClBuffer
{
public:
    using data_type = TData;
    using iterator = std::vector<TData>::iterator;
    using const_iterator = std::vector<TData>::const_iterator;

    CClBuffer() = default;
    CClBuffer([[maybe_unused]] const cl::CommandQueue& cmdQueue, size_t size) :
        _buf(size)
#ifdef ANN_GPU_CALCULATIONS
        , _cmdQueue(&cmdQueue), _bufCl(cmdQueue, _buf.begin(), _buf.end(), false)
#endif // ANN_GPU_CALCULATIONS
    {}
    CClBuffer(const CClBuffer<TData>& copy) : _buf(copy._buf)
#ifdef ANN_GPU_CALCULATIONS
        , _cmdQueue(copy._cmdQueue), _bufCl(copy._bufCl)
#endif // ANN_GPU_CALCULATIONS
    {}

    CClBuffer<TData>& CopyFromDevice() noexcept { GPU_Action(assert(_cmdQueue)); GPU_Action(cl::copy(*_cmdQueue, _bufCl, begin(), end())); return *this; }
    CClBuffer<TData>& CopyToDevice() noexcept   { GPU_Action(assert(_cmdQueue)); GPU_Action(cl::copy(*_cmdQueue, begin(), end(), _bufCl)); return *this; }

    std::vector<TData>& buffer() noexcept       { return _buf; }
    iterator begin() noexcept                   { return _buf.begin(); }
    iterator end() noexcept                     { return _buf.end();   }
    data_type* data() noexcept                  { return _buf.data();  }
    CClBuffer<TData>& fill(TData data) noexcept { std::fill(_buf.begin(), _buf.end(), data); return *this; }

    const std::vector<TData>& buffer() const noexcept { return _buf; }
    offset_type size() const noexcept      { return Convert<size_t, offset_type>(_buf.size()); }
    const_iterator begin() const noexcept  { return _buf.cbegin(); }
    const_iterator end() const noexcept    { return _buf.cend();   }
    const data_type* data() const noexcept { return _buf.data();   }

    cl::Buffer& clBuffer() noexcept                             { return _bufCl; }
    const cl::Buffer& clBuffer() const noexcept                 { return _bufCl; }

    CClBuffer& operator= (const std::vector<TData>& data)       { assert(_buf.size() == data.size()); _buf = data; return *this; }
    operator std::vector<TData>& () noexcept                    { return buffer(); }
    operator cl::Buffer& () noexcept                            { return clBuffer(); }
    operator const cl::Buffer&() const noexcept                 { return clBuffer(); }
    operator const std::vector<TData>& () const noexcept        { return buffer(); }
    operator data_type*() noexcept                              { return data(); }
    operator const data_type*() const noexcept                  { return data(); }
    data_type& operator[](size_t off)                           { assert(off < size()); return *(data() + off); }

private:
    std::vector<TData> _buf;
    const cl::CommandQueue* _cmdQueue = nullptr;
    cl::Buffer _bufCl;
};  // class CClBuffer

template <typename... TArgs>
class CKernelFunction
{
public:
    CKernelFunction() {}

#ifdef ANN_GPU_CALCULATIONS
    template<typename T> struct bufferSelector                      { using type = T; };
    template<typename T> struct bufferSelector<CClBuffer<T>&>       { using type = cl::Buffer; };
    template<typename T> struct bufferSelector<const CClBuffer<T>&> { using type = const cl::Buffer; };
    template<typename T> using bufferSelectorT = typename bufferSelector<T>::type;
    using TGPUFunctor = cl::KernelFunctor<bufferSelectorT<TArgs>...>;
    CKernelFunction(const cl::Program& program, cl::CommandQueue& cmdQueue, const std::string& kernelFunctionName) :
        _gpuKernelFunctor(std::make_unique<TGPUFunctor>(program, kernelFunctionName, nullptr)), _cmdQueue(&cmdQueue)
    {}
#else // ANN_GPU_CALCULATIONS
    template<typename T> struct bufferSelector          { using type = T; };
    template<> struct bufferSelector<cl::Buffer>        { using type = void*; };
    template<> struct bufferSelector<const cl::Buffer>  { using type = const void*; };
    template<typename T> using bufferSelectorT = typename bufferSelector<T>::type;
    using TCPUFunctor = std::function<void(bufferSelectorT<TArgs>...)>;
    CKernelFunction(TCPUFunctor kernelFunction) :
        _cpuKernelFunctor(kernelFunction)
    {}
#endif // ANN_GPU_CALCULATIONS

    cl::Event operator() ([[maybe_unused]] cl::Event waitEvent, [[maybe_unused]] size_t globalRange, [[maybe_unused]] cl_int& error, TArgs... args) const
    {
#ifdef ANN_GPU_CALCULATIONS
        assert(_cmdQueue);
        assert(_gpuKernelFunctor);
        return (*_gpuKernelFunctor)(cl::EnqueueArgs(*_cmdQueue, cl::NDRange(globalRange)), args..., error);
#else // ANN_GPU_CALCULATIONS
        assert(_cpuKernelFunctor);
        {
            std::lock_guard<std::mutex> lockMutex(openClEmulator::global_objects_mutex);
            openClEmulator::global_size[0] = globalRange;
            for (openClEmulator::global_id[0] = 0; openClEmulator::global_id[0] < globalRange; ++openClEmulator::global_id[0])
                _cpuKernelFunctor(args...);
        }
        return cl::Event();

#endif // ANN_GPU_CALCULATIONS
    }

private:
#ifdef ANN_GPU_CALCULATIONS
    cl::CommandQueue* _cmdQueue = nullptr;
    std::unique_ptr<TGPUFunctor> _gpuKernelFunctor;
#else // ANN_GPU_CALCULATIONS
    TCPUFunctor _cpuKernelFunctor;
#endif // ANN_GPU_CALCULATIONS

};  // class CKernelFunction

}   // namespace artificial_neural_network

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_INTERNALS_
