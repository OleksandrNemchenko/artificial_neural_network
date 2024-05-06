
#include <cassert>
#include <source_location>
#include <stacktrace>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <artificial_neural_network_internal/utilities.hpp>

#include "netCalculationImpl.hpp"
#include "openCl/openClCodeSrc.hpp"

using namespace artificial_neural_network;
using namespace std::string_literals;
using namespace nlohmann;

namespace artificial_neural_network {

#define checkCondition(SUCCESS_CONDITION, EXCEPTION, ERROR_MESSAGE)                     \
    do {                                                                                \
    if (!(SUCCESS_CONDITION))                                                           \
    {                                                                                   \
        assert(false);                                                                  \
        throw (EXCEPTION)(std::string(ERROR_MESSAGE) +                                  \
            "\nStack trace: "s + std::to_string(std::stacktrace::current()) + "\n"s +   \
            std::string(std::source_location::current().function_name()) + ":"s +       \
            std::to_string(std::source_location::current().line()));                    \
    }                                                                                   \
    }                                                                                   \
    while(0)

/* static */ net_calculation::net_calculation_inst net_calculation::make(const net_structure& network, std::string_view gpu_device)
{
    return std::make_unique<CNetCalculationImpl>(network, gpu_device);
}
CNetCalculationImpl::CNetCalculationImpl(const net_structure& network, std::string_view gpu_device) :
    _netStructure(static_cast<const CNetStructureImpl&>(network))
{
    std::string gpuDevice(gpu_device);
    _initOpenClEnvironment = std::async(std::launch::async, [this, gpuDevice]() { InitOpenCL(gpuDevice); });
}

CNetCalculationImpl::CNetCalculationImpl(const net_structure& network, cl::Device device, cl::Context context, cl::CommandQueue cmdQueue) :
    _netStructure(static_cast<const CNetStructureImpl&>(network)), _device(device), _context(context), _cmdQueue(cmdQueue)
{
    _initOpenClEnvironment = std::async(std::launch::async, [this]() { InitOpenCL(); });
}

void CNetCalculationImpl::InitOpenCL(const std::string gpuDevice)
{
#ifdef ANN_GPU_CALCULATIONS
    assert(!_device.get());
    assert(!_context.get());
    assert(!_cmdQueue.get());

    _device = FindDevice(gpuDevice);
    _context = cl::Context(_device);
    _cmdQueue = cl::CommandQueue(_context, _device);
#endif // ANN_GPU_CALCULATIONS

    InitOpenCL();
}

void CNetCalculationImpl::InitOpenCL()
{
#ifdef ANN_GPU_CALCULATIONS
    checkCondition(_device.get(),   std::runtime_error, "OpenCL's device is not initialized");
    checkCondition(_context.get(),  std::runtime_error, "OpenCL's context is not initialized");
    checkCondition(_cmdQueue.get(), std::runtime_error, "OpenCL's command queue is not initialized");

    _clProg = cl::Program(_context, _openClCode);

    try
    {
        std::string buildOptions = "-cl-std=CL3.0 -DANN_OPENCL_CODE";
        _clProg.build(buildOptions.c_str());
    }
    catch (...)
    {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = _clProg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        std::string errDescr;
        for (auto& pair : buildInfo)
            errDescr += (errDescr.empty() ? ""s : ", "s) + pair.second;
        
        _lastError = errDescr;

        assert(false);
        throw std::runtime_error("Error while building openCL code"s + (errDescr.empty() ? ""s : "\n"s + errDescr));
    }
#endif // ANN_GPU_CALCULATIONS

    _neuronsConfigs = CClBuffer<neuron_config_type>(_cmdQueue, _netStructure.configs_array_size());

    _inputOffsets = CClBuffer<offset_type>(_cmdQueue, _netStructure._inputsOff.size());
    _inputOffsets = _netStructure._inputsOff;
    _inputOffsets.CopyToDevice();

    _neurons = CClBuffer<SNeuron>(_cmdQueue, _netStructure.neurons_amount());
    _neurons = _netStructure._neurons;
    _neurons.CopyToDevice();

#ifdef ANN_GPU_CALCULATIONS
    _calculateNet = FCalculateNets(_clProg, _cmdQueue, "NetCalculations");
#else //ANN_GPU_CALCULATIONS
    _calculateNet = FCalculateNets(openClEmulator::NetCalculations );
#endif // ANN_GPU_CALCULATIONS

}

void CNetCalculationImpl::set_config(const ext_data_array& config)
{
    checkCondition(config.size() == _netStructure.configs_array_size(), std::logic_error, "Provided config size "s + std::to_string(config.size()) + " is not equal to the expected one "s + std::to_string(_netStructure.configs_array_size()));
    assert(_neuronsConfigs.size() == _netStructure.configs_array_size());

    for (size_t i = 0; i < config.size(); ++i)
        _neuronsConfigs[i] = Convert<ext_data_array::value_type, neuron_config_type>(config[i]);
    _neuronsConfigs.CopyToDevice();
}

ext_data_arrays CNetCalculationImpl::calculate(const ext_data_arrays& inputsSets, const ext_data_array& config)
{
    set_config(config);
    return calculate(inputsSets);
}

ext_data_arrays CNetCalculationImpl::calculate(const ext_data_arrays& inputsSets)
{
    checkCondition(initialized(), std::logic_error, "Calculation has to be initialized prior to call it. Wait until initialized() returns true"s);
    checkCondition(last_error().empty(), std::logic_error, "Check for error code"s);

    const offset_type inputsSetsAmount = Convert<size_t, offset_type>(inputsSets.size());
    CClBuffer<neuron_state_type> neuronsStates(_cmdQueue, _netStructure._statesSize * inputsSetsAmount);

    _netStructure.SetInputs(inputsSets, neuronsStates);

    for (size_t i = 1; i < _netStructure._layers.size(); ++i)
    {
        cl::Event event;

        const CNetStructureImpl::SLayer& layer = _netStructure._layers.at(i);
        cl_int clError = 0;
        
        static constexpr offset_type netConfigsAmount = 1;
        event = _calculateNet(event, layer._amount * inputsSetsAmount, clError,
            _neuronsConfigs, _inputOffsets, _neurons, neuronsStates, inputsSetsAmount, layer._first, layer._amount,
            Convert<size_t, offset_type>(_netStructure.neurons_amount()), netConfigsAmount, Convert<size_t, offset_type>(_netStructure.configs_array_size()));
        CheckClError(clError, "Unable to calculate network");

        WaitEvent(event);
        neuronsStates.CopyFromDevice();
    }

    return _netStructure.Outputs(neuronsStates);
}

net_calculation::calculation_results CNetCalculationImpl::calculate(const ext_data_arrays& inputsSets, const ext_data_arrays& configsSets)
{
    return calculate(inputsSets, configsSets, _calculateNet);
}

net_calculation::calculation_results CNetCalculationImpl::calculate(const ext_data_arrays& inputsSets, const ext_data_arrays& configsSets, FCalculateNets& calcNets)
{
    checkCondition(!configsSets.empty(), std::logic_error, "Configurations amount has not to be empty"s);
    checkCondition(!inputsSets.empty(), std::logic_error, "Configurations amount has not to be empty"s);

    const size_t configsSetsAmount = configsSets.size();
    const size_t inputsSetsAmount = inputsSets.size();

    const size_t configsAmountPerNet = _netStructure.configs_array_size();
    const size_t inputsAmountPerNet = _netStructure.inputs_amount();

    for (const auto& configs : configsSets)
        checkCondition(configs.size() == configsAmountPerNet, std::logic_error, "Provided config size "s + std::to_string(configs.size()) + " is not equal to the expected one "s + std::to_string(_netStructure.configs_array_size()));
    for (const auto& inputs : inputsSets)
        checkCondition(inputs.size() == inputsAmountPerNet, std::logic_error, "Provided inputs size "s + std::to_string(inputs.size()) + " is not equal to the expected one "s + std::to_string(_netStructure.inputs_amount()));

    const size_t configNetsAmount = inputsSetsAmount * configsSetsAmount;
    CClBuffer<neuron_config_type> neuronsConfigs(_cmdQueue, _netStructure.configs_array_size() * configNetsAmount);
    size_t off = 0;
    for (size_t i = 0; i < configsSetsAmount; ++i)
        for (size_t j = 0; j < _netStructure.configs_array_size(); ++j)
            neuronsConfigs[off++] = Convert<ext_data_array::value_type, neuron_config_type>(configsSets[i][j]);
    neuronsConfigs.CopyToDevice();

    CClBuffer<neuron_state_type> neuronsStates(_cmdQueue, _netStructure._statesSize * configNetsAmount);
    _netStructure.SetInputs(inputsSets, neuronsStates);

    for (size_t i = 1; i < _netStructure._layers.size(); ++i)
    {
        cl::Event event;

        const CNetStructureImpl::SLayer& layer = _netStructure._layers.at(i);
        cl_int clError = 0;
        
        event = calcNets(event, layer._amount * configNetsAmount, clError,
            neuronsConfigs, _inputOffsets, _neurons, neuronsStates, Convert<size_t, offset_type>(inputsSetsAmount), layer._first, layer._amount,
            Convert<size_t, offset_type>(_netStructure.neurons_amount()), Convert<size_t, offset_type>(configsSetsAmount), Convert<size_t, offset_type>(_netStructure.configs_array_size()));
        CheckClError(clError, "Unable to calculate network");

        WaitEvent(event);
        neuronsStates.CopyFromDevice();
    }

    ext_data_arrays outputsSet = _netStructure.Outputs(neuronsStates);

    net_calculation::calculation_results results;
    results.reserve(outputsSet.size());

    size_t inputOff = 0;
    size_t configOff = 0;
    for (ext_data_array& output : outputsSet)
    {
        calculation_result calcResult;

        calcResult._inputOff = inputOff++;
        calcResult._configOff = configOff;
        calcResult._output = std::move(output);

        results.emplace_back(std::move(calcResult));

        if (inputOff == inputsSetsAmount)
        {
            inputOff = 0;
            ++configOff;
        }
    }

    assert(inputOff == 0 && configOff == configsSetsAmount);

    return results;
}

/* static */ ext_data_array net_calculation::make_net_configs(const net_structure& network, const nlohmann::json& array)
{
    ext_data_array config;
    checkCondition(array.size() == network.configs_array_size(), std::logic_error, "Provided config size "s + std::to_string(network.configs_array_size()) + " is not equal to the expected one "s + std::to_string(array.size()));
    config.reserve(network.configs_array_size());

    for (const auto& configValue : array)
    {
        if (configValue.is_number())
            config.emplace_back(configValue);
        else if (configValue.is_string())
            config.emplace_back(ActFunctCode(configValue));
    }

    return config;
}

}   // namespace artificial_neural_network
