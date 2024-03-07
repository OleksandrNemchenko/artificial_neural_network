
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
#define _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_

#include <cassert>
#include <string>
#include <vector>

#include <annMetaConfigurations.hpp>
#include <halfFloat.hpp>

namespace artificial_neural_network
{

using ext_data_type = double;
using ext_data_array = std::vector<ext_data_type>;
using ext_data_arrays = std::vector<ext_data_array>;

// OPENCL CODE BEGINNING

typedef enum {
    BEGIN_PERCEPTRON_ACTIVATION_FUNCTIONS = 0,
    IDENTITY = BEGIN_PERCEPTRON_ACTIVATION_FUNCTIONS, SIGMOID, BINARY_STEP, TANH, RELU, SOFTPLUS, ELU, SELU, LRELU, PRELU, SILU, GAUSSIAN, /* SOFTMAX */
    END_PERCEPTRON_ACTIVATION_FUNCTIONS, PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT = END_PERCEPTRON_ACTIVATION_FUNCTIONS,
    INPUT = PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT, PERCEPTRON
} activation_function;

inline activation_function activation_function_from_net_config(double x)
{
    size_t preActFunct = (size_t)((x < 0 ? -x : x) * 100);
    preActFunct %= PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT;
    activation_function actFunct = (activation_function)preActFunct;

    return actFunct;
}
inline double activation_function_to_net_config(activation_function x)
{
    double actFunctCode = (double)x / 100.0;
    return actFunctCode;
}

#ifdef ANN_OPENCL_CODE

#if ANN_NEURON_STATE_TYPE_BYTES == 2 || ANN_NEURON_CONFIG_TYPE_BYTES == 2 || ANN_FITNESS_FUNCT_TYPE_BYTES == 2 || ANN_INTERPRET_DATA_TYPE_BYTES == 2
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // half support

#define activation_function_type uchar

#if ANN_NEURON_STATE_TYPE_BYTES == 1
    typedef char neuron_state_type;
#elif ANN_NEURON_STATE_TYPE_BYTES == 2
    typedef half neuron_state_type;
#elif ANN_NEURON_STATE_TYPE_BYTES == 4
    typedef float neuron_state_type;
#elif ANN_NEURON_STATE_TYPE_BYTES == 8
    typedef double neuron_state_type;
#endif // ANN_NEURON_STATE_TYPE_BYTES

#if ANN_NEURON_CONFIG_TYPE_BYTES == 1
    typedef uchar neuron_config_type;
    typedef uchar neuron_config_int_type;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 2
    typedef half neuron_config_type;
    typedef ushort neuron_config_int_type;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 4
    typedef float neuron_config_type;
    typedef uint neuron_config_int_type;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 8
    typedef double neuron_config_type;
    typedef ulong neuron_config_int_type;
#endif // ANN_NEURON_CONFIG_TYPE_BYTES

#if ANN_OFFSET_TYPE_BYTES == 1
    typedef uchar offset_type;
#elif ANN_OFFSET_TYPE_BYTES == 2
    typedef ushort offset_type;
#elif ANN_OFFSET_TYPE_BYTES == 4
    typedef uint offset_type;
#elif ANN_OFFSET_TYPE_BYTES == 8
    typedef ulong offset_type;
#endif // ANN_OFFSET_TYPE_BYTES

#endif // ANN_OPENCL_CODE

// OPENCL CODE ENDING

using activation_function_type = uint8_t;

#if ANN_NEURON_STATE_TYPE_BYTES == 1
    using neuron_state_type = char;
#elif ANN_NEURON_STATE_TYPE_BYTES == 2
    using neuron_state_type = float16;
#elif ANN_NEURON_STATE_TYPE_BYTES == 4
    using neuron_state_type = float;
#elif ANN_NEURON_STATE_TYPE_BYTES == 8
    using neuron_state_type = double;
#endif // ANN_NEURON_STATE_TYPE_BYTES

#if ANN_NEURON_CONFIG_TYPE_BYTES == 1
    using neuron_config_type = char;
    using neuron_config_int_type = uint8_t;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 2
    using neuron_config_type = float16;
    using neuron_config_int_type = uint16_t;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 4
    using neuron_config_type = float;
    using neuron_config_int_type = uint32_t;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 8
    using neuron_config_type = double;
    using neuron_config_int_type = uint64_t;
#endif // ANN_NEURON_CONFIG_TYPE_BYTES

#if ANN_OFFSET_TYPE_BYTES == 1
    using offset_type = uint8_t;
#elif ANN_OFFSET_TYPE_BYTES == 2
    using offset_type = uint16_t;
#elif ANN_OFFSET_TYPE_BYTES == 4
    using offset_type = uint32_t;
#elif ANN_OFFSET_TYPE_BYTES == 8
    using offset_type = uint64_t;
#endif // ANN_OFFSET_TYPE_BYTES

const std::string& ActFunct(activation_function actFunct);
activation_function ActFunct(const std::string& actFunct);
inline double ActFunctCode(const std::string& actFunct)
{
    return activation_function_to_net_config(ActFunct(actFunct));
}

}   // namespace artificial_neural_network

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
