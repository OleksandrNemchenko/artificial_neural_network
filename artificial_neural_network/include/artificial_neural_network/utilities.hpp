
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
#define _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_

#include <string>
#include <type_traits>
#include <vector>

#include <halfFloat.hpp>
#include <annDataTypes.hpp>

namespace artificial_neural_network
{

#if ANN_INPUT_TYPE_BYTES == 1
    using input_type = char;
#elif ANN_INPUT_TYPE_BYTES == 2
    using input_type = float16;
#elif ANN_INPUT_TYPE_BYTES == 4
    using input_type = float;
#elif ANN_INPUT_TYPE_BYTES == 8
    using input_type = double;
#endif // ANN_INPUT_TYPE_BYTES

#if ANN_OUTPUT_TYPE_BYTES == 1
    using output_type = char;
#elif ANN_OUTPUT_TYPE_BYTES == 2
    using output_type = float16;
#elif ANN_OUTPUT_TYPE_BYTES == 4
    using output_type = float;
#elif ANN_OUTPUT_TYPE_BYTES == 8
    using output_type = double;
#endif // ANN_OUTPUT_TYPE_BYTES

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
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 2
    using neuron_config_type = float16;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 4
    using neuron_config_type = float;
#elif ANN_NEURON_CONFIG_TYPE_BYTES == 8
    using neuron_config_type = double;
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

template<typename TSrc, typename TDst>
inline TDst Convert(const TSrc value)
{
    using namespace std::string_literals;

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

}   // namespace artificial_neural_network

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
