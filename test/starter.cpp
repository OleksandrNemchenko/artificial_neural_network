
#include <functional>
#include <iostream>
#include <source_location>
#include <string>

#include "tests.hpp"

using namespace std::string_literals;

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    try
    {
        CheckUtilities();
        CheckCalculations();
    }
    catch (std::runtime_error err)
    {
        std::cout << "* Error happened: "s << err.what() << std::endl;
        return -1;
    }

    return 0;
}