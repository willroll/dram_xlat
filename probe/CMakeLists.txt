cmake_minimum_required(VERSION 3.13)
project(dram_probe C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(dram_probe src/probe.c src/platform.c)
target_compile_options(dram_probe PRIVATE
  -O2 -Wall -Wextra -march=native)

# Build on the target so -march=native selects clflushopt/rdtscp correctly.
