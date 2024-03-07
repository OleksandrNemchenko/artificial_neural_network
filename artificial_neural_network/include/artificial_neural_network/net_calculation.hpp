
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_
#define _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/utilities.hpp>

namespace artificial_neural_network
{

class net_calculation
{
public:
    struct calculation_result
    {
        size_t _inputOff;
        size_t _configOff;
        ext_data_array _output;
    };

    using calculation_results = std::vector<calculation_result>;
    using net_calculation_inst = std::unique_ptr<net_calculation>;

    static net_calculation_inst make(const net_structure& network, std::string_view gpu_device = "");
    static net_calculation_inst make(const std::unique_ptr<net_structure>& network, std::string_view gpu_device = "") { return make(*network.get(), gpu_device); }
    virtual ~net_calculation() noexcept = default;

    virtual size_t inputs_amount() const noexcept = 0;
    virtual size_t outputs_amount() const noexcept = 0;
    virtual size_t configs_array_size() const noexcept = 0;
    virtual bool initialized() const noexcept = 0;
    virtual const std::string& last_error() const noexcept = 0;

    virtual void set_config(const ext_data_array& config) = 0;
    virtual ext_data_array calculate(const ext_data_array& inputsSets) { return calculate(ext_data_arrays({ inputsSets })).at(0); }
    virtual ext_data_array calculate(const ext_data_array& inputsSets, const ext_data_array& config) { return calculate(ext_data_arrays({ inputsSets }), config).at(0); }
    virtual ext_data_arrays calculate(const ext_data_arrays& inputs) = 0;
    virtual ext_data_arrays calculate(const ext_data_arrays& inputsSets, const ext_data_array& configsSets) = 0;
    virtual calculation_results calculate(const ext_data_arrays& inputs, const ext_data_arrays& config) = 0;

    static ext_data_array make_net_configs(const std::unique_ptr<net_structure>& network, const nlohmann::json& array) { return make_net_configs(*network.get(), array); }
    static ext_data_array make_net_configs(const net_structure& network, const nlohmann::json& array);
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_
