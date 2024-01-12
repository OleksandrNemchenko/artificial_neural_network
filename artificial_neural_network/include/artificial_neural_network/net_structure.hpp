
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
#define _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_

#include <memory>

namespace artificial_neural_network
{

class net_structure
{
public:
    enum ESourceType { INPUTS, NEURONS, AUTO };
    enum EConnectionType { FULLY_CONNECTED, P2P_CONNECTED };

    static std::unique_ptr<net_structure> Make(size_t inputsAmount, size_t outputsAmount);
    static std::unique_ptr<net_structure> Make(const std::unique_ptr<net_structure>& network);
    virtual ~net_structure() noexcept = default;

    virtual size_t InputsAmount() const noexcept = 0;
    virtual size_t OutputsAmount() const noexcept = 0;
    virtual size_t NeuronsAmount() const noexcept = 0;
    virtual size_t CurLayerNeuronsAmount() const noexcept = 0;

    void AddP2PNeuronsLayer()                               { AddP2PNeuronsLayer(ESourceType::AUTO); }
    void AddFullyConnectedNeuronsLayer()                    { AddFullyConnectedNeuronsLayer(ESourceType::AUTO); }
    void AddFullyConnectedNeuronsLayer(size_t neuronsToAdd) { AddFullyConnectedNeuronsLayer(neuronsToAdd, ESourceType::AUTO);  }
    virtual void AddP2PNeuronsLayer(ESourceType sourceType) = 0;
    virtual void AddFullyConnectedNeuronsLayer(ESourceType sourceType) = 0;
    virtual void AddFullyConnectedNeuronsLayer(size_t neuronsToAdd, ESourceType sourceType) = 0;
    virtual void AddNeuronsLayer(ESourceType sourceType, EConnectionType connectionType) = 0;
    virtual void AddNeuronsLayer(ESourceType sourceType, EConnectionType connectionType, size_t neuronsToAdd) = 0;

    void AddOutputLayer() { return AddOutputLayer(ESourceType::AUTO, EConnectionType::FULLY_CONNECTED);  }
    void AddOutputLayer(EConnectionType connectionType) { return AddOutputLayer(ESourceType::AUTO, connectionType);  }
    virtual void AddOutputLayer(ESourceType sourceType, EConnectionType connectionType) = 0;
    virtual void SetLastLayerAsOutput() = 0;
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_
