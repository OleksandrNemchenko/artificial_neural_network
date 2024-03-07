
#ifndef _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_
#define _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_

#include <vector>

#include <artificial_neural_network/net_calculation.hpp>
#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/utilities.hpp>
#include <artificial_neural_network_internal/utilities.hpp>
#include <nlohmann/json.hpp>
#include <annMetaConfigurations.hpp>

namespace artificial_neural_network
{

class CNetStructureImpl : public net_structure
{
    friend class CNetCalculationImpl;
    friend class CGeneticAlgorithm;
    friend class CBackPropagation;
    friend class CDeepLearningAlgorithm;

public:
    CNetStructureImpl(const net_structure& network);
    CNetStructureImpl(const nlohmann::json& network);
    ~CNetStructureImpl() noexcept override = default;

    nlohmann::json export_json() const noexcept override;

    size_t inputs_amount() const noexcept override  { return _inputs; }
    size_t outputs_amount() const noexcept override { return _outputs; }
    size_t neurons_amount() const noexcept override { return _neurons.size(); }
    size_t configs_array_size() const noexcept override { return _configsSize; }

private:
    using offset_type = offset_type;
    using EActivationFunction = activation_function;
    using TInputsOff = std::vector<offset_type>;

    using TNeurons = std::vector<SNeuron>;

    struct SLayer
    {
        offset_type _first;
        offset_type _amount;
    };
    using TLayers = std::vector<SLayer>;

    struct SLayerBuilder : public SLayer
    {
        static activation_function ActivationFunction(const nlohmann::json& settings) noexcept;
        void InitLayer(CNetStructureImpl* insnetStructt, size_t amount);
        TNeurons::iterator AllocateNeurons(CNetStructureImpl* insnetStructt, size_t amount);
        TInputsOff::iterator AllocateInputs(CNetStructureImpl* insnetStructt, size_t amount);
    };

    using TLayersBuilder = std::vector<SLayerBuilder>;

    size_t _inputs;
    size_t _outputs;
    size_t _configsSize;
    size_t _statesSize = 0;
    TNeurons _neurons;
    TLayersBuilder _layers;
    TInputsOff _inputsOff;
    size_t _firstOutputState;

    void AddInputLayer();
    void AddPointToPointLayer(const nlohmann::json& settings);
    void AddFullyConnectedLayer(const nlohmann::json& settings);

    void SetInputs(const ext_data_arrays& inputs, CClBuffer<neuron_state_type>& neuronsStates) const;
    ext_data_arrays Outputs(CClBuffer<neuron_state_type>& neuronsStates) const;
    size_t OutputsOffset() const;
};

}   // namespace artificialNeuralNetwork

#endif // _ARTIFICAL_NEURAL_NETWORK_NET_STRUCTURE_IMPL_