cmake_minimum_required(VERSION 3.25)
project(EngineZero CXX)

add_executable(AI
        src/main.cpp
        src/Tensor.cpp
        src/Ops.cpp
        src/Autograd.cpp
)

target_include_directories(AI PRIVATE include)

if(MSVC)
    set_target_properties(AI PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
    )

    target_compile_options(AI PRIVATE /std:c++latest /arch:AVX2)

    target_compile_options(AI PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2 /fp:fast>)
else()
    set_target_properties(AI PROPERTIES
            CXX_STANDARD 26
            CXX_STANDARD_REQUIRED ON
    )
    target_compile_options(AI PRIVATE -march=native)
    target_compile_options(AI PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3 -ffast-math>)
    target_compile_options(AI PRIVATE -fopenmp)
    target_link_libraries(AI PRIVATE gomp)
endif()