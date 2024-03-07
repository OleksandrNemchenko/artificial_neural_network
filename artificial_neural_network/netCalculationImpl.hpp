
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_IMPL_
#define _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_IMPL_

#include <chrono>
#include <future>
#include <string>
#include <vector>

#include <CL/opencl.hpp>

#include <nlohmann/json.hpp>

#include <annMetaConfigurations.hpp>
#include <artificial_neural_network/net_calculation.hpp>
#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/utilities.hpp>
#include <artificial_neural_network_internal/utilities.hpp>

#include "netStructureImpl.hpp"
#include "openCl/openClCodeSrc.hpp"

namespace artificial_neural_network
{

class CNetCalculationImpl : public net_calculation
{
    friend class CGeneticAlgorithm;
    friend class CBackPropagation;

public:
    CNetCalculationImpl(const net_structure& network, std::string_view gpu_device);
    CNetCalculationImpl(const net_structure& network, cl::Device device, cl::Context context, cl::CommandQueue cmdQueue);
    ~CNetCalculationImpl() noexcept override = default;

    size_t inputs_amount() const noexcept override  { return _netStructure._inputs; }
    size_t outputs_amount() const noexcept override { return _netStructure._outputs; }
    size_t configs_array_size() const noexcept override { return _netStructure._configsSize; }
    bool initialized() const noexcept override { return _initOpenClEnvironment.wait_for(std::chrono::seconds{ 0 }) == std::future_status::ready; }
    const std::string& last_error() const noexcept override { return _lastError; }

    void set_config(const ext_data_array& config) override;
    ext_data_arrays calculate(const ext_data_arrays& inputsSets) override;
    ext_data_arrays calculate(const ext_data_arrays& inputsSets, const ext_data_array& config) override;
    calculation_results calculate(const ext_data_arrays& inputsSets, const ext_data_arrays& configsSets) override;

private:
    using FCalculateNets = CKernelFunction<
        const CClBuffer<neuron_config_type>&, const CClBuffer<offset_type>&, const CClBuffer<SNeuron>&, CClBuffer<neuron_state_type>&,
        offset_type, offset_type, offset_type, offset_type, offset_type, offset_type>;

    static const std::vector<std::string> _openClCode;

    const CNetStructureImpl& _netStructure;
    std::future<void> _initOpenClEnvironment;
    std::string _lastError;

    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _cmdQueue;
    cl::Program _clProg;
    cl::Event _waitEvent;
    CClBuffer<neuron_config_type> _neuronsConfigs;
    CClBuffer<offset_type> _inputOffsets;
    CClBuffer<SNeuron> _neurons;
    FCalculateNets _calculateNet;

    calculation_results calculate(const ext_data_arrays& inputsSets, const ext_data_arrays& configsSets, FCalculateNets& calcNets);

    void InitOpenCL();
    void InitOpenCL(const std::string gpuDevice);
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_CALCULATION_IMPL_