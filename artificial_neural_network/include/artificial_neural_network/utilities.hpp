
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
#define _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_

#include <vector>

#include <halfFloat.hpp>

namespace artificial_neural_network
{

using data_type = float16;
using offset_type = size_t;

using datas_type = std::vector<data_type>;
using offsets_type = std::vector<offset_type>;

}   // namespace artificial_neural_network

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_UTILITIES_
