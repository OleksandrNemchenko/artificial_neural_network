
#include <iostream>
#include <string>
#include <artificial_neural_network/utilities.hpp>

#include "test_tools.hpp"

using namespace std::string_literals;
using namespace artificial_neural_network;

void CheckUtilities()
{
    std::cout << "- Test utilities";

    Test([]() { return ActFunct(activation_function::IDENTITY)    == "identity"s    && ActFunct("identity"s)    == activation_function::IDENTITY;    });
    Test([]() { return ActFunct(activation_function::SIGMOID)     == "sigmoid"s     && ActFunct("sigmoid"s)     == activation_function::SIGMOID;     });
    Test([]() { return ActFunct(activation_function::BINARY_STEP) == "binary_step"s && ActFunct("binary_step"s) == activation_function::BINARY_STEP; });
    Test([]() { return ActFunct(activation_function::TANH)        == "tanh"s        && ActFunct("tanh"s)        == activation_function::TANH;        });
    Test([]() { return ActFunct(activation_function::RELU)        == "relu"s        && ActFunct("relu"s)        == activation_function::RELU;        });
    Test([]() { return ActFunct(activation_function::GELU)        == "gelu"s        && ActFunct("gelu"s)        == activation_function::GELU;        });
    Test([]() { return ActFunct(activation_function::SOFTPLUS)    == "softplus"s    && ActFunct("softplus"s)    == activation_function::SOFTPLUS;    });
    Test([]() { return ActFunct(activation_function::ELU)         == "elu"s         && ActFunct("elu"s)         == activation_function::ELU;         });
    Test([]() { return ActFunct(activation_function::SELU)        == "selu"s        && ActFunct("selu"s)        == activation_function::SELU;        });
    Test([]() { return ActFunct(activation_function::LRELU)       == "lrelu"s       && ActFunct("lrelu"s)       == activation_function::LRELU;       });
    Test([]() { return ActFunct(activation_function::PRELU)       == "prelu"s       && ActFunct("prelu"s)       == activation_function::PRELU;       });
    Test([]() { return ActFunct(activation_function::SILU)        == "silu"s        && ActFunct("silu"s)        == activation_function::SILU;        });
    Test([]() { return ActFunct(activation_function::GAUSSIAN)    == "gaussian"s    && ActFunct("gaussian"s)    == activation_function::GAUSSIAN;    });
    Test([]() { return ActFunct(activation_function::SOFTMAX)     == "softmax"s     && ActFunct("softmax"s)     == activation_function::SOFTMAX;     });

    bool err;
    try { err = false; ActFunct(activation_function::PERCEPTRON_ACTIVATION_FUNCTIONS_AMOUNT); } catch (std::out_of_range) { err = true; } Test([err](){ return err; });
    try { err = false; ActFunct("activation_functions_amount"s                  ); } catch (std::out_of_range) { err = true; } Test([err]() { return err; });

    std::cout << std::endl;

}
