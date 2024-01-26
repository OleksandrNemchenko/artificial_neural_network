
#include <cassert>
#include <source_location>
#include <stacktrace>
#include <stdexcept>
#include <string>

#include "netStructureImpl.hpp"

using namespace artificial_neural_network;
using namespace std::string_literals;

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

/* static */ const std::unordered_map<CNetStructureImpl::EActivationFunction, std::string> CNetStructureImpl::_activationFunctStr =
{
    { UNSPECIFIED,       "unspecified"s    },
    { IDENTITY,          "identity"s       },
    { SIGMOID,           "sigmoid"s        },
    { BINARY_STEP,       "binary_step"s    },
    { BINARY_STEP_PARAM, "binary_step"s    },
    { IDENTITY_PARAM,    "identity_param"s },
    { TANH,              "tanh"s           },
    { RELU,              "relu"s           },
    { RELU_PARAM,        "relu_param ram"s },
    { SOFTPLUS,          "softplus"s       },
    { ELU,               "elu"s            },
    { ELU_PARAM,         "elu_param"s      },
    { SELU,              "selu"s           },
    { LRELU,             "lrelu"s          },
    { SILU,              "silu"s           },
    { GAUSSIAN,          "gaussian"s       },
    { SOFTMAX,           "softmax"s        }
};

/* static */ std::unique_ptr<net_structure> net_structure::Make(size_t inputsAmount, size_t outputsAmount)
{
    return std::make_unique<CNetStructureImpl>(inputsAmount, outputsAmount);
}
CNetStructureImpl::CNetStructureImpl(size_t inputsAmount, size_t outputsAmount):
    _inputs(inputsAmount), _outputs(outputsAmount)
{
    assert(CNetStructureImpl::_activationFunctStr.size() == CNetStructureImpl::EActivationFunction::ACTIVATION_FUNCTIONS_AMOUNT);

    checkCondition(inputsAmount, std::out_of_range, "inputs amount has to be more than zero");
    checkCondition(outputsAmount, std::out_of_range, "outputs amount has to be more than zero");
}

/* static */ std::unique_ptr<net_structure> net_structure::Make(const std::unique_ptr<net_structure>& network)
{
    return std::make_unique<CNetStructureImpl>(network);
}
CNetStructureImpl::CNetStructureImpl(const std::unique_ptr<net_structure>& network):
    _inputs(static_cast<const CNetStructureImpl&>(*network)._inputs), _outputs(static_cast<const CNetStructureImpl&>(*network)._outputs)
{
    const CNetStructureImpl& net = static_cast<const CNetStructureImpl&>(*network);

    _neurons = net._neurons;
    _layers = net._layers;
    _inputsOff = net._inputsOff;
    _configsSize = net._configsSize;
    _statesSize = net._statesSize;
}

/* static */ std::unique_ptr<net_structure> net_structure::Make(const nlohmann::json& network)
{
    return std::make_unique<CNetStructureImpl>(network);
}
CNetStructureImpl::CNetStructureImpl(const nlohmann::json& network)
{
    const nlohmann::json& settings = network.at("data version 1");

    _inputs = settings.at("inputs");
    _outputs = settings.at("outputs");
    _configsSize = settings.at("configs size");
    _statesSize = settings.at("states size");

    _inputsOff.reserve(settings.at("inputs offsets").size());
    for (const auto& off : settings.at("inputs offsets"))
        _inputsOff.emplace_back(off);

    _layers.reserve(settings.at("layers").size());
    for (const nlohmann::json& layer : settings.at("layers"))
    {
        SRange range;

        range._first = layer.at("first");
        range._amount = layer.at("amount");

        _layers.emplace_back(std::move(range));
    }

    _neurons.reserve(settings.at("neurons").size());
    for (const nlohmann::json& neuronJson : settings.at("neurons"))
    {
        SNeuron neuron;

        for (const auto& actFunct : _activationFunctStr)
            if (actFunct.second == neuronJson.at("activation function").get<std::string>())
                neuron._activationFunction = actFunct.first;

        neuron._layerNeuronPosition = neuronJson.at("layer neuron position");
        neuron._inputsAmount = neuronJson.at("inputs amount");
        neuron._firstInputOff = neuronJson.at("first input offset");
        neuron._firstConfigOff = neuronJson.at("first config offset");
        neuron._stateOff = neuronJson.at("state offset");

        _neurons.emplace_back(std::move(neuron));
    }
}

size_t CNetStructureImpl::CurLayerNeuronsAmount() const noexcept
{
    if (_layers.empty())
        return InputsAmount();

    return (_layers.end() - 1)->_amount;
}

void CNetStructureImpl::AddP2PNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType)
{
    AddNeuronsLayer(activationFunction, sourceType, EConnectionType::P2P_CONNECTED);
}

void CNetStructureImpl::AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType)
{
    AddNeuronsLayer(activationFunction, sourceType, EConnectionType::FULLY_CONNECTED);
}

void CNetStructureImpl::AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, size_t neuronsToAdd, ESourceType sourceType)
{
    AddNeuronsLayer(activationFunction, sourceType, EConnectionType::FULLY_CONNECTED, neuronsToAdd);
}

void CNetStructureImpl::AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType)
{
    AddNeuronsLayer(activationFunction, sourceType, connectionType, CurLayerNeuronsAmount());
}

void CNetStructureImpl::AddOutputLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType)
{
    AddNeuronsLayer(activationFunction, sourceType, connectionType, OutputsAmount());
    SetLastLayerAsOutput();
}

void CNetStructureImpl::AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType, size_t neuronsToAdd)
{
    checkCondition(!_layers.empty() || (sourceType == INPUTS || sourceType == AUTO), std::logic_error, "first layer has not to be connected to neurons");

    if (sourceType == AUTO)
        sourceType = _layers.empty() ? INPUTS : NEURONS;

    size_t prevLayerSize = CurLayerNeuronsAmount();

    SRange newLayerRange;
    newLayerRange._first = _neurons.size();
    newLayerRange._amount = neuronsToAdd;
    _layers.emplace_back(std::move(newLayerRange));

    size_t inputsAmount = 0;
    switch (connectionType)
    {
    case FULLY_CONNECTED : inputsAmount = prevLayerSize; break;
    case P2P_CONNECTED:    inputsAmount = 1; break;
    default:    assert(false);
    }

    const size_t prevNeuronsAmount = _neurons.size();
    _neurons.reserve(prevNeuronsAmount + neuronsToAdd);
    _inputsOff.reserve(_inputsOff.size() + neuronsToAdd * inputsAmount);

    size_t firstStateAsInputOff = 0;
    switch (sourceType)
    {
    case INPUTS:  firstStateAsInputOff = 0; break;
    case NEURONS: firstStateAsInputOff = _neurons.size() - prevLayerSize; break;
    default:    assert(false);
    }

    for (size_t i = 0; i < neuronsToAdd; ++i)
    {
        SNeuron neuron;

        neuron._activationFunction = activationFunction;
        neuron._layerNeuronPosition = i;
        neuron._inputsAmount = static_cast<decltype(SNeuron::_inputsAmount)>(inputsAmount);
        neuron._firstInputOff = static_cast<decltype(SNeuron::_firstInputOff)>(_inputsOff.size());

        for (size_t j = 0; j < inputsAmount; ++j)
        {
            TOffset inputOff = static_cast<TOffset>(firstStateAsInputOff + j);
            assert(firstStateAsInputOff + j < _externalDirBit);

            if (sourceType == INPUTS)
                inputOff |= _externalDirBit;

            _inputsOff.emplace_back(inputOff);
        }

        if (connectionType == P2P_CONNECTED)
            ++firstStateAsInputOff;

        neuron._firstConfigOff = static_cast<decltype(SNeuron::_firstConfigOff)>(_configsSize);
        _configsSize += inputsAmount + _configsPerNeuron;

        neuron._stateOff = static_cast<decltype(SNeuron::_stateOff)>(_statesSize);
        ++_statesSize;

        _neurons.emplace_back(std::move(neuron));
    }
}

void CNetStructureImpl::SetLastLayerAsOutput()
{
    checkCondition(!_layers.empty(), std::logic_error, "at least one neurons layer has to be created before this call");
    checkCondition(CurLayerNeuronsAmount() == _outputs, std::out_of_range, "last neuron layer "s + std::to_string(_layers.size()) + " has to have "s + std::to_string(_outputs) + " neurons that is equal to outputs amount whereas it has "s + std::to_string(CurLayerNeuronsAmount()) + " neurons"s);

    for (size_t i = _neurons.size() - _outputs, j = 0; i < _neurons.size(); ++i, ++j)
        _neurons[i]._stateOff = _externalDirBit | j;

    _statesSize -= _outputs;
}

size_t CNetStructureImpl::ActFunctOffset(size_t neuronPos) const
{
    checkCondition(neuronPos < _neurons.size(), std::out_of_range, "neuron position "s + std::to_string(neuronPos) + " has to be less that the neurons amount "s + std::to_string(_neurons.size()));

    const SNeuron& neuron = _neurons.at(neuronPos);
    return neuron._firstConfigOff + neuron._inputsAmount;
}

size_t CNetStructureImpl::ActFunctParam1Offset(size_t neuronPos) const
{
    return ActFunctOffset(neuronPos) + 1;
}

size_t CNetStructureImpl::ActFunctParam2Offset(size_t neuronPos) const
{
    return ActFunctParam1Offset(neuronPos) + 1;
}

size_t CNetStructureImpl::WeightOffset(size_t neuronPos, size_t inputPos) const
{
    checkCondition(neuronPos < _neurons.size(), std::out_of_range, "neuron position "s + std::to_string(neuronPos) + " has to be less that the neurons amount "s + std::to_string(_neurons.size()));

    const SNeuron& neuron = _neurons.at(neuronPos);
    checkCondition(inputPos < neuron._inputsAmount, std::out_of_range, "input position "s + std::to_string(inputPos) + " for neuron "s + std::to_string(neuronPos) + " has to be less that the inputs amount "s + std::to_string(neuron._inputsAmount) + " for this neuron"s);

    return neuron._firstConfigOff + inputPos;
}

nlohmann::json CNetStructureImpl::Export() const noexcept
{
    nlohmann::json generalResult;
    nlohmann::json& result = generalResult["data version 1"];

    nlohmann::json& neurons = result["neurons"];
    for (const SNeuron& neuron : _neurons)
    {
        nlohmann::json neuronJson;

        neuronJson["activation function"] = _activationFunctStr.at(neuron._activationFunction);
        neuronJson["layer neuron position"] = neuron._layerNeuronPosition;
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
}
