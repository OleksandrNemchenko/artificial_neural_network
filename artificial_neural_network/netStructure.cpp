
#include <cassert>
#include <source_location>
#include <stacktrace>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <artificial_neural_network_internal/utilities.hpp>

#include "netCalculationImpl.hpp"
#include "netStructureImpl.hpp"

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

/* static */ net_structure::net_structure_inst net_structure::make(const net_structure& network)
{
    return std::make_unique<CNetStructureImpl>(network);
}
CNetStructureImpl::CNetStructureImpl(const net_structure& network)
{
    const CNetStructureImpl& net = static_cast<const CNetStructureImpl&>(network);

    _inputs = net._inputs;
    _inputsOff = net._inputsOff;
    _outputs = net._outputs;
    _configsSize = net._configsSize;
    _statesSize = net._statesSize;
    _neurons = net._neurons;
    _layers = net._layers;
}

/* static */ net_structure::net_structure_inst net_structure::make(const nlohmann::json& network)
{
    return std::make_unique<CNetStructureImpl>(network);
}
CNetStructureImpl::CNetStructureImpl(const nlohmann::json& network)
{
    checkCondition(network.contains("data version 1"), std::logic_error, "Network description has to contain \"data version 1\" root element");
    const json& rootSettings = network.at("data version 1");

    checkCondition(rootSettings.contains("inputs"), std::logic_error, "Configuration has to contain \"inputs\" element with inputs amount");
    _inputs = rootSettings.at("inputs");

    checkCondition(rootSettings.contains("layers"), std::logic_error, "Configuration has to contain \"layers\" element with layers information");
    const json& layersSettings = rootSettings["layers"];

    _layers.reserve(1 /* inputs amount */ + layersSettings.size());

    AddInputLayer();
    for (const json& layerSettings : layersSettings)
    {
        checkCondition(layerSettings.contains("type"), std::logic_error, "Layer description has to contain \"type\" field with layer type information");
        const std::string type = layerSettings["type"];

        if (type == "point to point")       AddPointToPointLayer(layerSettings);
        else if (type == "fully connected") AddFullyConnectedLayer(layerSettings);
        else                                checkCondition(false, std::logic_error, "Unexpected network layer type "s + type);
    }

    _firstOutputState = _layers.back()._first;
    _outputs = _layers.back()._amount;
}

void CNetStructureImpl::AddInputLayer()
{
    SLayerBuilder layer;

    layer.InitLayer(this, _inputs);
    auto firstNeuron = layer.AllocateNeurons(this, _inputs);

    for (size_t i = 0; i < _inputs; ++i)
    {
        SNeuron& neuron = _neurons[i];

        neuron._neuronType = activation_function::INPUT;
        neuron._inputsAmount = 0;
        neuron._firstInputOff = 0;
        neuron._firstConfigOff = 0;
    }

    _layers.emplace_back(std::move(layer));
}

void CNetStructureImpl::AddPointToPointLayer(const nlohmann::json& settings)
{
    SLayerBuilder layer;

    size_t inputs = _layers.back()._amount;
    size_t firstPrevNeuron = _layers.back()._first;

    layer.InitLayer(this, inputs);
    auto firstNeuron = layer.AllocateNeurons(this, inputs);
    auto inputOffIt = layer.AllocateInputs(this, inputs);
    size_t prevNeuron = firstPrevNeuron;
    for (; inputOffIt != _inputsOff.end(); ++inputOffIt, ++prevNeuron)
        *inputOffIt = _neurons[prevNeuron]._stateOff;

    activation_function actFunct = SLayerBuilder::ActivationFunction(settings);
    offset_type layerNeuronPosition = 0;

    const offset_type dConfigSize = (actFunct == activation_function::PERCEPTRON ? 1 /* act. function */ : 0) + 1 /* w1 */ + 1 /* w0 */;

    prevNeuron = firstPrevNeuron;
    for (auto neuronIt = firstNeuron; neuronIt != _neurons.end(); )
    {
        neuronIt->_neuronType = static_cast<activation_function_type>(actFunct);
        neuronIt->_inputsAmount = 1;
        neuronIt->_firstInputOff = Convert<size_t, offset_type>(prevNeuron);
        neuronIt->_firstConfigOff = Convert<size_t, offset_type>(_configsSize);

        ++neuronIt;
        ++layerNeuronPosition;
        ++prevNeuron;
        _configsSize += dConfigSize;
    }

    _layers.emplace_back(std::move(layer));
}

void CNetStructureImpl::AddFullyConnectedLayer(const nlohmann::json& settings)
{
    SLayerBuilder layer;

    offset_type inputs = _layers.back()._amount;
    size_t neurons = settings.contains("neurons") ? settings["neurons"].get<int>() : inputs;

    offset_type prevLayerFirstInputOff = Convert<size_t, offset_type>(_layers.back()._first);

    layer.InitLayer(this, neurons);
    auto firstNeuron = layer.AllocateNeurons(this, neurons);
    auto inputOffIt = layer.AllocateInputs(this, inputs);

    activation_function actFunct = SLayerBuilder::ActivationFunction(settings);
    offset_type layerNeuronPosition = 0;

    const offset_type dConfigSize = (actFunct == activation_function::PERCEPTRON ? 1 /* act. function */ : 0) + 1 /* w0 */;

    for (offset_type i = 0; i < inputs; ++i, ++inputOffIt)
        *inputOffIt = prevLayerFirstInputOff + i;

    for (auto neuronIt = firstNeuron; neuronIt != _neurons.end(); )
    {
        neuronIt->_neuronType = static_cast<activation_function_type>(actFunct);
        neuronIt->_inputsAmount = inputs;
        neuronIt->_firstInputOff = prevLayerFirstInputOff;
        neuronIt->_firstConfigOff = Convert<size_t, offset_type>(_configsSize);

        ++neuronIt;
        ++layerNeuronPosition;
        _configsSize += inputs * 1 /* wi */ + dConfigSize;
    }

    _layers.emplace_back(std::move(layer));
}

void CNetStructureImpl::SLayerBuilder::InitLayer(CNetStructureImpl* netStruct, size_t amount)
{
    assert(netStruct);

    _first = Convert<size_t, offset_type>(netStruct->_neurons.size());
    _amount = Convert<size_t, offset_type>(amount);
}

/* static */ activation_function CNetStructureImpl::SLayerBuilder::ActivationFunction(const nlohmann::json& settings) noexcept
{
    activation_function actFunct = activation_function::PERCEPTRON;

    if (settings.contains("activation function"))
        actFunct = ActFunct(settings["activation function"].get<std::string>());

    return actFunct;
}

CNetStructureImpl::TNeurons::iterator CNetStructureImpl::SLayerBuilder::AllocateNeurons(CNetStructureImpl* netStruct, size_t amount)
{
    assert(netStruct);

    auto& neurons = netStruct->_neurons;
    size_t prevSize = neurons.size();
    neurons.resize(prevSize + amount);
    TNeurons::iterator firstNeuron = neurons.end() - amount;

    for (size_t i = prevSize; i < neurons.size(); ++i, ++netStruct->_statesSize)
    {
        neurons[i]._stateOff = Convert<size_t, offset_type>(netStruct->_statesSize);
//        neurons[i]._layerNeuronPosition = i - prevSize;   TODO: for softmax
    }

    return firstNeuron;
}

CNetStructureImpl::TInputsOff::iterator CNetStructureImpl::SLayerBuilder::AllocateInputs(CNetStructureImpl* netStruct, size_t amount)
{
    assert(netStruct);

    netStruct->_inputsOff.resize(netStruct->_inputsOff.size() + amount);
    TInputsOff::iterator firstInputOff = netStruct->_inputsOff.end() - amount;

    return firstInputOff;
}

nlohmann::json CNetStructureImpl::export_json() const noexcept
{
    json generalResult;
//    json& result = generalResult["data version 1"];
//
//    result["inputs"] = _inputs;
//
//    json& layers = result["layers"];
//    for (size_t i = 1 /* skip input layer */; i < _layers.size(); ++i)
//    {
//        // TODO
//    }

    return generalResult;
    /*
        nlohmann::json generalResult;
        nlohmann::json& result = generalResult["data version 1"];

        nlohmann::json& neurons = result["neurons"];
        for (const SNeuron& neuron : _neurons)
        {
            nlohmann::json neuronJson;

            neuronJson["activation function"] = _activationFunctStr.at(neuron._activationFunction);
//            neuronJson["layer neuron position"] = neuron._layerNeuronPosition;    TODO: for softmax
            neuronJson["inputs amount"] = neuron._inputsAmount;
            neuronJson["first input offset"] = neuron._firstInputOff;
            neuronJson["first config offset"] = neuron._firstConfigOff;
            neuronJson["state offset"] = neuron._stateOff;

            neurons.emplace_back(std::move(neuronJson));
        }

        nlohmann::json& layers = result["layers"];
        for (const SRange& layer : _layers)
        {
            nlohmann::json layerJson;

            layerJson["first"] = layer._first;
            layerJson["amount"] = layer._amount;

            layers.emplace_back(std::move(layerJson));
        }

        result["inputs offsets"] = _inputsOff;
        result["configs size"] = _configsSize;
        result["states size"] = _statesSize;
        result["inputs"] = _inputs;
        result["outputs"] = _outputs;

        return generalResult;
    */
}

void CNetStructureImpl::SetInputs(const ext_data_arrays& inputsSets, CClBuffer<neuron_state_type>& neuronsStates) const
{
    size_t inputSetsAmount = inputsSets.size();
    const size_t neuronsAmount = neurons_amount();
    checkCondition(inputSetsAmount > 0, std::logic_error, "Inputs sets amount has not to be zero"s);
    checkCondition(neuronsStates.size() % (inputSetsAmount * neuronsAmount) == 0, std::logic_error, "Neurons states amount has to be correct"s);

    auto neuronsStatesIt = neuronsStates.begin();

    const size_t neuronsStatesSetsAmount = neuronsStates.size() / neuronsAmount / inputSetsAmount;

    for (size_t i = 0; i < neuronsStatesSetsAmount; ++i)
    {
        for (size_t j = 0; j < inputSetsAmount; ++j)
        {
            checkCondition(inputsSets.at(j).size() == inputs_amount(), std::logic_error, "Provided inputs amount "s + std::to_string(inputsSets.at(j).size()) + " has to be the same as network structure inputs "s + std::to_string(inputs_amount()) + " one"s);
            for (size_t k = 0; k < inputs_amount(); ++k)
                *(neuronsStatesIt + k) = Convert<ext_data_array::value_type, neuron_state_type>(inputsSets.at(j).at(k));

            if (static_cast<size_t>(neuronsStates.end() - neuronsStatesIt) > neuronsAmount)
                neuronsStatesIt += neuronsAmount;
        }
    }

    neuronsStates.CopyToDevice();
}

ext_data_arrays CNetStructureImpl::Outputs(CClBuffer<neuron_state_type>& neuronsStates) const
{
    checkCondition(neuronsStates.size() % neurons_amount() == 0, std::logic_error, "Neurons sets amount has to be correct"s);
    const size_t setsAmount = neuronsStates.size() / neurons_amount();

    neuronsStates.CopyFromDevice();
    
    ext_data_arrays outputs(setsAmount);

    auto neuronsStatesIt = neuronsStates.begin() + neurons_amount() - outputs_amount();
    for (size_t i = 0; i < setsAmount; ++i)
    {
        outputs[i].resize(outputs_amount());
        auto pDst = outputs[i].begin();

        for (size_t j = 0; j < outputs_amount(); ++j, ++pDst)
            *pDst = Convert<neuron_state_type, ext_data_array::value_type>(*(neuronsStatesIt + j));

        if (i < (setsAmount - 1))
            neuronsStatesIt += neurons_amount();
    }

    return outputs;
}

size_t CNetStructureImpl::OutputsOffset() const
{
    const size_t outputOffset = neurons_amount() - outputs_amount();

    return outputOffset;
}

}   // namespace artificial_neural_network
