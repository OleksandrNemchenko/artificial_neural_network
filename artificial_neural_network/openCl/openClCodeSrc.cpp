
#include <annMetaConfigurations.hpp>

#include <numbers>

#ifndef ANN_GPU_CALCULATIONS

#include <artificial_neural_network_internal/utilities.hpp>
#include <artificial_neural_network_internal/openClHidings.hpp>

#include "openClCodeSrc.hpp"

namespace artificial_neural_network::openClEmulator
{

// OPENCL CODE BEGINNING

__global const neuron_config_type* ProcessNeuron
(
    __global const SNeuron* neuron, __global const offset_type* inputsOffs,
    __global neuron_state_type* neuronsStates, __global const neuron_config_type* config
)
{
//    const bool isSoftMax = neuron->_neuronType == SOFTMAX;

    offset_type inputOffPtr = neuron->_firstInputOff;
//    double softMaxNumerator = 0;
    config += neuron->_firstConfigOff;

    activation_function actFunct = (activation_function)neuron->_neuronType;
    if (actFunct == PERCEPTRON)
        actFunct = activation_function_from_net_config(*config++);

    double sum = *config++;

    for (offset_type i = 0; i < neuron->_inputsAmount; ++i, ++inputOffPtr, ++config)
    {
        const offset_type inputOff = inputsOffs[inputOffPtr];
        const double inputValue = neuronsStates[inputOff];
        const double curInputValue = inputValue * *config;
//        const double curInputValue = isSoftMax ? exp(inputValue) : (inputValue * *config);
        sum += curInputValue;

//        if (isSoftMax && i == neuron->_layerNeuronPosition)
//            softMaxNumerator = exp(inputValue);
    }

    const double e_plus_sum = exp(sum);
    const double e_minus_sum = exp(-sum);

    static const double alpha = 1.67326;
    static const double lambda = 1.0507;

    double res;
    switch (actFunct)
    {
    case IDENTITY:          res = sum; break;
    case SIGMOID:           res = 1 / (1 + e_minus_sum); break;
    case BINARY_STEP:       res = (sum < 0 ? 0 : 1); break;
    case TANH:              res = (e_plus_sum - e_minus_sum) / (e_plus_sum + e_minus_sum); break;
    case RELU:              res = (sum <= 0 ? 0 : sum); break;
    case SOFTPLUS:          res = log(1 + e_plus_sum); break;
    case ELU:               res = (sum < 0 ? alpha * (e_plus_sum - 1) : sum); break;
    case SELU:              res = lambda * (sum < 0 ? alpha * (e_plus_sum - 1) : sum); break;
    case LRELU:             res = (sum < 0 ? 0.01 * sum : sum); break;
    case PRELU:             res = (sum < 0 ? alpha * sum : sum); break;
    case SILU:              res = sum / (1 + e_minus_sum); break;
    case GAUSSIAN:          res = exp(-(sum * sum)); break;
//    case SOFTMAX:           res = softMaxNumerator / sum; break;
    default:                res = -1.7976931348623157e+308;
    }

    if (!isnormal(res))
        res = 0;

    const offset_type resOff = neuron->_stateOff;
    neuronsStates[resOff] = (neuron_state_type)res;

    return config;
}

__kernel void NetCalculations
(
    __global const void* netConfigsPtr, __global const void* inputsOffsPtr, __global const void* neuronsPtr, __global void* neuronsStatesPtr,
    const offset_type inputsSetsAmount, const offset_type firstNeuron, const offset_type neuronsAmount, const offset_type netNeuronsAmount, const offset_type netConfigsAmount, const offset_type netConfigsPerNet
)
{
    const offset_type inputsSet = Index1of3(get_global_id(0), inputsSetsAmount);
    const offset_type neuron = firstNeuron + Index2of3(get_global_id(0), inputsSetsAmount, neuronsAmount);
    const offset_type netConfig = Index3of3(get_global_id(0), inputsSetsAmount, neuronsAmount);
    if (neuron - firstNeuron > neuronsAmount || inputsSet > inputsSetsAmount || netConfig > netConfigsAmount)
        return;

    __global const neuron_config_type* netConfigs = (__global const neuron_config_type*)netConfigsPtr + netConfig * netConfigsPerNet;
    __global const offset_type* inputsOffs = (__global const offset_type*)inputsOffsPtr;
    __global const SNeuron* neurons = (__global const SNeuron*)neuronsPtr + neuron;
    __global neuron_state_type* neuronsStates = (__global neuron_state_type*)neuronsStatesPtr + (inputsSet + netConfig * inputsSetsAmount ) * netNeuronsAmount;

    ProcessNeuron(neurons, inputsOffs, neuronsStates, netConfigs);
}

// OPENCL CODE ENDING

}   // namespace artificial_neural_network::openClEmulator

#endif // ANN_GPU_CALCULATIONS
