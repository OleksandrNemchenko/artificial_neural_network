
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include <artificial_neural_network/net_calculation.hpp>
#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/utilities.hpp>

#include "test_tools.hpp"
#include "tests.hpp"

using namespace std::string_literals;
using namespace artificial_neural_network;
using namespace nlohmann;

void CheckConfig(const json& testConfig)
{
#ifdef _DEBUG
    const std::string jsonStr = testConfig.dump(4);
#endif // _DEBUG

    std::cout << "- Test " << testConfig["title"].get<std::string>();

    auto netStruct = net_structure::make(testConfig["structure"]);
    auto netCalc = net_calculation::make(netStruct);

    ext_data_array config = net_calculation::make_net_configs(netStruct, testConfig["net configurations"]);

    struct STest
    {
        ext_data_array _inputs;
        ext_data_array _expectedOutputs;
        ext_data_array _actualOutputs;
        long double _allowedError;
    };
    std::vector<STest> tests;

    ext_data_arrays inputs;
    for (const auto& testSample : testConfig["test samples"])
    {
#ifdef _DEBUG
        const std::string testSampleStr = testSample.dump(4);
#endif // _DEBUG

        STest test;

        test._inputs = testSample["inputs"].get<std::vector<ext_data_type>>();
        test._expectedOutputs = testSample["outputs"].get<std::vector<ext_data_type>>();
        test._allowedError = testSample["error"];

        inputs.emplace_back(test._inputs);
        tests.emplace_back(std::move(test));
    }

    while (!netCalc->initialized())
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

    auto correctTest = [](const STest& test, const ext_data_array& actualOutputs)
    {
        for (size_t i = 0; i < test._expectedOutputs.size(); ++i)
            if (std::abs(test._expectedOutputs[i] - actualOutputs[i]) > test._allowedError)
                return false;

        return true;
    };

    for (const STest& test : tests)
    {
        const ext_data_array actualOutputs = netCalc->calculate(test._inputs, config);
        Test([&test, &actualOutputs, &correctTest]() { return correctTest(test, actualOutputs); });
    }

    ext_data_arrays outputs = netCalc->calculate(inputs, config);
    for (size_t i = 0; i < tests.size(); ++i)
    {
        const STest& test = tests[i];
        const ext_data_array& actualOutputs = outputs[i];
        Test([&test, &actualOutputs, &correctTest]() { return correctTest(test, actualOutputs); });
    }

    std::cout << std::endl;
}

void CheckCalculations()
{
    for (const std::string& testConfig : _testConfigs)
        CheckConfig(json::parse(testConfig));

    auto feedForward1 = json::parse(_testConfigs[0]);
    auto feedForward3 = json::parse(_testConfigs[2]);

    auto netStruct = net_structure::make(feedForward1["structure"]);
    auto netCalc = net_calculation::make(netStruct);

    while (!netCalc->initialized())
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

    ext_data_arrays inputs;
    inputs.emplace_back(feedForward1["test samples"][0]["inputs"].get<ext_data_array>());
    inputs.emplace_back(feedForward1["test samples"][1]["inputs"].get<ext_data_array>());
    inputs.emplace_back(feedForward1["test samples"][2]["inputs"].get<ext_data_array>());
    inputs.emplace_back(feedForward1["test samples"][3]["inputs"].get<ext_data_array>());

    ext_data_arrays netConfigs;

    ext_data_array config;
    netConfigs.emplace_back(net_calculation::make_net_configs(netStruct, feedForward1["net configurations"]));
    netConfigs.emplace_back(net_calculation::make_net_configs(netStruct, feedForward3["net configurations"]));

    auto actualResults = netCalc->calculate(inputs, netConfigs);

    auto correctTest = [](const ext_data_array& expectedResults, const ext_data_array& actualOutputs, long double allowedError)
    {
        for (size_t i = 0; i < expectedResults.size(); ++i)
            if (std::abs(expectedResults[i] - actualOutputs[i]) > allowedError)
                return false;

        return true;
    };

    size_t i = 0;
    Test([&]() { return correctTest(feedForward1["test samples"][0]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward1["test samples"][0]["error"]); });
    Test([&]() { return correctTest(feedForward1["test samples"][1]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward1["test samples"][1]["error"]); });
    Test([&]() { return correctTest(feedForward1["test samples"][2]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward1["test samples"][2]["error"]); });
    Test([&]() { return correctTest(feedForward1["test samples"][3]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward1["test samples"][3]["error"]); });
    Test([&]() { return correctTest(feedForward3["test samples"][0]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward3["test samples"][0]["error"]); });
    Test([&]() { return correctTest(feedForward3["test samples"][1]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward3["test samples"][1]["error"]); });
    Test([&]() { return correctTest(feedForward3["test samples"][2]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward3["test samples"][2]["error"]); });
    Test([&]() { return correctTest(feedForward3["test samples"][3]["outputs"].get<ext_data_array>(), actualResults[i++]._output, feedForward3["test samples"][3]["error"]); });

}
