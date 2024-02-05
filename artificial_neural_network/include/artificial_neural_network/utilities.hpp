
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
#define _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_

#include <string>
#include <type_traits>
#include <vector>

#include <halfFloat.hpp>

namespace artificial_neural_network
{

using data_type = float16;
using offset_type = uint32_t;

using datas_type = std::vector<data_type>;
using offsets_type = std::vector<offset_type>;

inline offset_type ConvertToOffsetType(const size_t offset)
{
    using namespace std::string_literals;

    if (offset > std::numeric_limits<offset_type>::max())
        throw std::runtime_error("Offset value "s + std::to_string(offset) + " is bigger than offset_type maximum value "s + std::to_string(std::numeric_limits<offset_type>::max()));

    return static_cast<offset_type>(offset);
}

}   // namespace artificial_neural_network

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
