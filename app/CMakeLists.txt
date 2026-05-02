cmake_minimum_required(VERSION 3.16)
project(offline_volume_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(offline_volume_app src/main.cpp)

# Placeholder only: hook up rendering stack (OpenGL/Vulkan/Qt) as needed.
