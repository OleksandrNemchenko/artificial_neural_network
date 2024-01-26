
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_
#define _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_

#include <vector>

#include <artificial_neural_network/net_structure.hpp>
#include <nlohmann/json.hpp>

namespace artificial_neural_network
{

class CNetStructureImpl : public net_structure
{
    friend class CCalcNetImpl;
    friend class CGeneticAlgorithmImpl;

public:
    CNetStructureImpl(size_t inputsAmount, size_t outputsAmount);
    CNetStructureImpl(const std::unique_ptr<net_structure>& network);
    CNetStructureImpl(const nlohmann::json& network);
    ~CNetStructureImpl() noexcept override = default;

    nlohmann::json Export() const noexcept override;

    size_t InputsAmount() const noexcept override  { return _inputs; }
    size_t OutputsAmount() const noexcept override { return _outputs; }
    size_t NeuronsAmount() const noexcept override { return _neurons.size(); }
    size_t CurLayerNeuronsAmount() const noexcept override;
    size_t ConfigsAmount() const noexcept { return _configsSize; }
    size_t StatesAmount() const noexcept { return _statesSize; }

    void AddP2PNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType) override;
    void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType) override;
    void AddFullyConnectedNeuronsLayer(EActivationFunction activationFunction, size_t neuronsToAdd, ESourceType sourceType) override;
    void AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType) override;
    void AddNeuronsLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType, size_t neuronsToAdd) override;
    void AddOutputLayer(EActivationFunction activationFunction, ESourceType sourceType, EConnectionType connectionType) override;
    void SetLastLayerAsOutput() override;

private:
    using TOffset = size_t;

    struct SNeuron
    {
        EActivationFunction _activationFunction;
        TOffset _layerNeuronPosition;
        TOffset _inputsAmount;
        TOffset _firstInputOff;
        TOffset _firstConfigOff;
        TOffset _stateOff;
    };

    struct SRange
    {
        TOffset _first;
        TOffset _amount;
    };

    static constexpr size_t _actFunctWordsAmount = 1;
    static constexpr size_t _actFunctParam1WordsAmount = 1;
    static constexpr size_t _actFunctParam2WordsAmount = 1;
    static constexpr size_t _configsPerNeuron = _actFunctWordsAmount + _actFunctParam1WordsAmount + _actFunctParam2WordsAmount;
    static constexpr TOffset _externalDirBit = static_cast<TOffset>(1LL << (sizeof(TOffset) * 8 - 1));
    static constexpr TOffset _offsetMask = _externalDirBit - 1;
    static const std::unordered_map<EActivationFunction, std::string> _activationFunctStr;
    size_t _inputs;
    size_t _outputs;

    std::vector<SNeuron> _neurons;
    std::vector<SRange> _layers;
    std::vector<TOffset> _inputsOff;
    size_t _configsSize = 0;
    size_t _statesSize = 0;

    size_t WeightOffset(size_t neuronPos, size_t inputPos) const;
    size_t ActFunctOffset(size_t neuronPos) const;
    size_t ActFunctParam1Offset(size_t neuronPos) const;
    size_t ActFunctParam2Offset(size_t neuronPos) const;
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_