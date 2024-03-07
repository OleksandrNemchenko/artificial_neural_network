
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
#define _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_

#include <memory>

#include <nlohmann/json.hpp>

namespace artificial_neural_network
{

class net_structure
{
public:
    using net_structure_inst = std::unique_ptr<net_structure>;
    static net_structure_inst make(const net_structure& network);
    static net_structure_inst make(const nlohmann::json& network);
    virtual ~net_structure() noexcept = default;

    virtual nlohmann::json export_json() const noexcept = 0;

    virtual size_t inputs_amount() const noexcept = 0;
    virtual size_t outputs_amount() const noexcept = 0;
    virtual size_t neurons_amount() const noexcept = 0;
    virtual size_t configs_array_size() const noexcept = 0;
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
