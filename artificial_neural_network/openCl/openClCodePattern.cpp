
#include <string>
#include "../netCalculationImpl.hpp"

namespace artificial_neural_network
{

/* static */ const std::vector<std::string> CNetCalculationImpl::_openClCode =
{
    R"(REPLACEMENT_ANN_META_CONFIGURATIONS)"s,
    R"(REPLACEMENT_UTILITIES_NOT_INTERNAL)"s,
    R"(REPLACEMENT_UTILITIES_INTERNALS)"s,
    R"(REPLACEMENT_OPENCL_PROGRAM_HEADER)"s,
    R"(REPLACEMENT_OPENCL_PROGRAM_CODE)"s,
};

} // namespace artificial_neural_network
