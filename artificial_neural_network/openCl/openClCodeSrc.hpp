
#ifndef ANN_GPU_CALCULATIONS

#ifndef _ANN_OPENCL_FUNCTIONS_HPP_
#define _ANN_OPENCL_FUNCTIONS_HPP_

#include <artificial_neural_network_internal/utilities.hpp>
#include <artificial_neural_network_internal/openClHidings.hpp>

namespace artificial_neural_network::openClEmulator
{

// OPENCL CODE BEGINNING

__global const neuron_config_type* ProcessNeuron
(
    __global const SNeuron* neuron, __global const offset_type* inputsOffs,
    __global neuron_state_type* neuronsStates, __global const neuron_config_type* config
);

__kernel void NetCalculations
(
    __global const void* netConfigs, __global const void* inputsOffs, __global const void* neurons, __global void* neuronsStates,
    const offset_type inputsSetsAmount, const offset_type firstNeuron, const offset_type neuronsAmount, const offset_type netNeuronsAmount, const offset_type netConfigsAmount, const offset_type netConfigsPerNet
);

// OPENCL CODE ENDING

}   // namespace artificial_neural_network::openClEmulator

#endif // _ANN_OPENCL_FUNCTIONS_HPP_

#endif // ANN_GPU_CALCULATIONS
