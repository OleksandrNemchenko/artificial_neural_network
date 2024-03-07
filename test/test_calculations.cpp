
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

    std::vector<long double> config;
    for (const auto& configValue : testConfig["net configurations"])
    {
        if (configValue.is_number())
            config.emplace_back(configValue);
        else if (configValue.is_string())
            config.emplace_back(ActFunctCode(configValue));
    }

    while (!netCalc->initialized())
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

    for (const auto& testSample : testConfig["test samples"])
    {
#ifdef _DEBUG
        const std::string testSampleStr = testSample.dump(4);
#endif // _DEBUG
        std::vector<long double> inputs = testSample["inputs"].get<std::vector<long double>>();
        
        const std::vector<long double>& expectedOutputs = testSample["outputs"].get<std::vector<long double>>();

        const std::vector<long double> actualOutputs = netCalc->calculate(inputs, config);

        const long double testError = testSample["error"];
        Test([&expectedOutputs, &actualOutputs, testError]()
        {
            if (expectedOutputs.size() != actualOutputs.size())
                return false;

            for (size_t i = 0; i < expectedOutputs.size(); ++i)
                if (std::abs(expectedOutputs[i] - actualOutputs[i]) > testError)
                    return false;

            return true;
        });
    }

    std::cout << std::endl;
}

void CheckCalculations()
{
    for (const std::string& testConfig : _testConfigs)
        CheckConfig(json::parse(testConfig));
}
