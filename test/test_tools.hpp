
#ifndef _ANN_TEST_TEST_TOOLS_HPP_
#define _ANN_TEST_TEST_TOOLS_HPP_

#include <cassert>
#include <functional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std::string_literals;

inline void Test(std::function<bool(void)> test, std::source_location src = std::source_location::current())
{
    if (!test())
    {
        assert(false);
        throw std::runtime_error("* Error test step on "s + src.file_name() + ", line "s + std::to_string(src.line()) + ", function "s + src.function_name());
    }
}

#endif // _ANN_TEST_TEST_TOOLS_HPP_
