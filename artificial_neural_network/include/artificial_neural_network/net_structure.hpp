
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
#define _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_

#include <memory>

#include <nlohmann/json.hpp>

namespace artificial_neural_network
{

class net_structure
{
public:
    enum ESourceType { INPUTS, NEURONS, AUTO };
    enum EConnectionType { FULLY_CONNECTED, P2P_CONNECTED };
    enum EActivationFunction : uint8_t { UNSPECIFIED = 0, IDENTITY, SIGMOID, BINARY_STEP_PARAM, BINARY_STEP, IDENTITY_PARAM, TANH, RELU, RELU_PARAM, SOFTPLUS, ELU, ELU_PARAM, SELU, LRELU, SILU, GAUSSIAN, SOFTMAX, ACTIVATION_FUNCTIONS_AMOUNT };

    static std::unique_ptr<net_structure> Make(size_t inputsAmount, size_t outputsAmount);
    static std::unique_ptr<net_structure> Make(const std::unique_ptr<net_structure>& network);
    static std::unique_ptr<net_structure> Make(const nlohmann::json& network);
    virtual ~net_structure() noexcept = default;

    virtual nlohmann::json Export() const noexcept = 0;

    virtual size_t InputsAmount() const noexcept = 0;
    virtual size_t OutputsAmount() const noexcept = 0;
    virtual size_t NeuronsAmount() const noexcept = 0;
    virtual size_t CurLayerNeuronsAmount() const noexcept = 0;

    void AddP2PNeuronsLayer()                                                       { AddP2PNeuronsLayer(EActivationFunction::UNSPECIFIED, ESourceType::AUTO); }
    void AddP2PNeuronsLayer(ESourceType sourceType)                                 { AddP2PNeuronsLayer(EActivationFunction::UNSPECIFIED, sourceType); }
    void AddFullyConnectedNeuronsLayer()                                            { AddFullyConnectedNeuronsLayer(EActivationFunction::UNSPECIFIED, ESourceType::AUTO); }
    void AddFullyConnectedNeuronsLayer(size_t neuronsToAdd)                         { AddFullyConnectedNeuronsLayer(EActivationFunction::UNSPECIFIED, neuronsToAdd, ESourceType::AUTO);  }
    void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction)      { AddFullyConnectedNeuronsLayer(activationFunction, ESourceType::AUTO); }
    void AddFullyConnectedNeuronsLayer(ESourceType sourceType)                      { AddFullyConnectedNeuronsLayer(EActivationFunction::UNSPECIFIED, sourceType); }
    void AddFullyConnectedNeuronsLayer(size_t neuronsToAdd, ESourceType sourceType)                 { AddFullyConnectedNeuronsLayer(EActivationFunction::UNSPECIFIED, neuronsToAdd, sourceType); }
    void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, size_t neuronsToAdd) { AddFullyConnectedNeuronsLayer(activationFunction, neuronsToAdd, ESourceType::AUTO); }
    void AddNeuronsLayer(ESourceType sourceType, EConnectionType connectionType)                        { AddNeuronsLayer(EActivationFunction::UNSPECIFIED, sourceType, connectionType); }
    void AddNeuronsLayer(ESourceType sourceType, EConnectionType connectionType, size_t neuronsToAdd)   { AddNeuronsLayer(EActivationFunction::UNSPECIFIED, sourceType, connectionType, neuronsToAdd); }
    void AddOutputLayer()                                                       { return AddOutputLayer(ESourceType::AUTO, EConnectionType::FULLY_CONNECTED);  }
    void AddOutputLayer(EActivationFunction activationFunction)                 { return AddOutputLayer(activationFunction, ESourceType::AUTO, EConnectionType::FULLY_CONNECTED); }
    void AddOutputLayer(ESourceType sourceType, EConnectionType connectionType) { return AddOutputLayer(EActivationFunction::UNSPECIFIED, sourceType, connectionType); }

    virtual void AddP2PNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType) = 0;
    virtual void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType) = 0;
    virtual void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, size_t neuronsToAdd, ESourceType sourceType) = 0;
    virtual void AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType) = 0;
    virtual void AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType, size_t neuronsToAdd) = 0;
    virtual void AddOutputLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType) = 0;
    virtual void SetLastLayerAsOutput() = 0;
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
