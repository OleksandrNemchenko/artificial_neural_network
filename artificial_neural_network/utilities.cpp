
#include <unordered_map>
#include <stdexcept>
#include <string>

#include <artificial_neural_network/utilities.hpp>
#include <artificial_neural_network_internal/utilities.hpp>

using namespace artificial_neural_network;
using namespace std::string_literals;

namespace
{

static const std::unordered_map<activation_function, std::string> _activationFunctMap
{
    { IDENTITY,          "identity"s       },
    { SIGMOID,           "sigmoid"s        },
    { BINARY_STEP,       "binary_step"s    },
    { TANH,              "tanh"s           },
    { RELU,              "relu"s           },
    { SOFTPLUS,          "softplus"s       },
    { ELU,               "elu"s            },
    { SELU,              "selu"s           },
    { LRELU,             "lrelu"s          },
    { PRELU,             "prelu"s          },
    { SILU,              "silu"s           },
    { GAUSSIAN,          "gaussian"s       },
//    { SOFTMAX,           "softmax"s        }
};

}   // namespace

namespace artificial_neural_network
{

#ifndef ANN_GPU_CALCULATIONS

namespace openClEmulator
{

std::mutex global_objects_mutex;
std::vector<size_t> global_id(3);       // global index
std::vector<size_t> global_size(3);     // global range
std::vector<size_t> local_id(3);        // local index within group
std::vector<size_t> local_size(3);      // group size
std::vector<size_t> num_groups(3);      // number of groups
std::vector<size_t> group_id(3);        // group ID
std::vector<size_t> global_offset(3);   // global offset

};  // namespace openClEmulator

#endif // ANN_GPU_CALCULATIONS

const std::string& artificial_neural_network::ActFunct(activation_function actFunct)
{
    assert(_activationFunctMap.size() == activation_function::PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT);

    return _activationFunctMap.at(actFunct);
}

activation_function artificial_neural_network::ActFunct(const std::string& actFunct)
{
    assert(_activationFunctMap.size() == activation_function::PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT);

    static const std::unordered_map<std::string, activation_function> reversedMap = []()
    {
        std::unordered_map<std::string, activation_function> reversedMapInit;

        for (const auto elem : _activationFunctMap)
            reversedMapInit[elem.second] = elem.first;

        return reversedMapInit;
    }();

    return reversedMap.at(actFunct);
}

#ifdef ANN_GPU_CALCULATIONS
cl::Device FindDevice(std::string_view deviceName)
{
    static const std::unordered_map<std::string, cl::Device> availableDevicesSt = []()
    {
        cl_int errCode;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty())
            throw std::runtime_error("There is no OpenCL platforms");

        std::unordered_map<std::string, cl::Device> availableDevices;

        for (const cl::Platform& platform : platforms)
        {
            std::string platformVer = platform.getInfo<CL_PLATFORM_VERSION>();
            if (platformVer.find("OpenCL 3") == std::string::npos)
                continue;

            std::vector<cl::Device> platformDevices;
            errCode = platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
            if (errCode != CL_SUCCESS)
                throw std::runtime_error("Error while gettings devices for the "s + platform.getInfo<CL_PLATFORM_NAME>() + " platform");
            for (cl::Device& device : platformDevices)
            {
                std::string str = device.getInfo<CL_DEVICE_NAME>();
                availableDevices[str] = device;
            }
        }

        return availableDevices;
    }();

    if (deviceName.empty())
    {
        size_t maxSpeed = 0;
        std::string quickedDeviceName;
        for (auto device = availableDevicesSt.begin(); device != availableDevicesSt.end(); ++device)
        {
            size_t deviceSpeed = device->second.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
            if (deviceSpeed > maxSpeed)
            {
                maxSpeed = deviceSpeed;
                quickedDeviceName = device->first;
            }
        }

        return availableDevicesSt.at(quickedDeviceName);
    }

    auto devIt = availableDevicesSt.find(deviceName.data());
    if (devIt != availableDevicesSt.end())
       return devIt->second;

    throw std::logic_error("No "s + deviceName.data() + " GPU device has been found"s);
}
#endif // ANN_GPU_CALCULATIONS

} // namespace artificial_neural_network
