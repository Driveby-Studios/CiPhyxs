#==================================================================================================
# CMakeLists.txt  —  CiPhyxs Physics Engine
#
# A high-performance, header-only C++20 physics engine for industrial/mechanical simulations.
#
# Usage:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Options:
#   CIPHYXS_BUILD_TESTS   ON/OFF (default: ON)
#   CIPHYXS_ENABLE_SIMD   ON/OFF (default: OFF — no compiler flags added)
#==================================================================================================

cmake_minimum_required(VERSION 3.16)
project(CiPhyxs
    VERSION       0.1.0
    DESCRIPTION   "Header-only C++20 physics engine for high-mass-ratio industrial simulation"
    LANGUAGES     CXX
)

# ─── C++ Standard ────────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ─────────────────────────────────────────────────────────────────────────────────────
option(CIPHYXS_BUILD_TESTS "Build the test suite" ON)
option(CIPHYXS_ENABLE_SIMD "Add compiler flags for SIMD (SSE2/NEON)" OFF)

# ─── Header-only interface library ───────────────────────────────────────────────────────────────

add_library(ciphyxs INTERFACE)

target_include_directories(ciphyxs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(ciphyxs INTERFACE cxx_std_20)

# ─── SIMD optimisations ─────────────────────────────────────────────────────────────────────────
if(CIPHYXS_ENABLE_SIMD)
  if(MSVC)
    target_compile_options(ciphyxs INTERFACE /arch:SSE2)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(ciphyxs INTERFACE -msse2 -msse3 -msse4.1)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    target_compile_options(ciphyxs INTERFACE -march=armv8-a+simd)
  endif()
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────────────────────────
if(CIPHYXS_BUILD_TESTS)
  # Prevent CMake from searching for a C compiler (all source is C++ only).
  enable_testing()

  set(TEST_SOURCES
        Tests/test_alignment.cpp
        Tests/test_collision.cpp
        Tests/test_final.cpp
        Tests/test_joints.cpp
        Tests/test_new_features.cpp
        Tests/test_parallel.cpp
        Tests/test_sap.cpp
        Tests/test_convexmesh.cpp
        Tests/test_trimesh.cpp
        Tests/test_ccd.cpp
        Tests/test_ragdoll.cpp
        Tests/test_dbvt.cpp
        Tests/test_contactreduction.cpp
        Tests/test_vehicle.cpp
        Tests/test_character.cpp
        Tests/test_taskgraph.cpp
        Tests/test_softbody.cpp
        Tests/test_fracture.cpp
        )

  foreach(src ${TEST_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})

    # MSVC uses /W4, /O2, /Od /Zi; GCC/Clang use -Wall -Wextra -Wpedantic, -O2, -g -O0.
    if(MSVC)
      target_compile_options(${name} PRIVATE /W4
                $<$<CONFIG:Release>:/O2 /DNDEBUG>
                $<$<CONFIG:Debug>:/Od /Zi>
            )
    else()
      target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic
          $<$<CONFIG:Release>:-O3>
          $<$<CONFIG:Debug>:-g -O0>
      )
    endif()

    target_link_libraries(${name} PRIVATE ciphyxs)
    # Add the project root so tests can use "include/ciphyxs/CiPhyxs.hpp" relative includes.
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    # Per-test timeouts for the regular test suite.
    # test_taskgraph runs 11+ subtests with 120-frame physics simulations.
    if(${name} STREQUAL "test_taskgraph")
      set_tests_properties(${name} PROPERTIES TIMEOUT 120)
    else()
      set_tests_properties(${name} PROPERTIES TIMEOUT 60)
    endif()
  endforeach()

  # ─── Stress-Test Suite ──────────────────────────────────────────────────────────────────────
  set(STRESS_TEST_SOURCES
        Stress_Tests/DeepStackTest.cpp
        Stress_Tests/TunnelingTest.cpp
        Stress_Tests/CCDConvexMeshTest.cpp
        Stress_Tests/ChaosDensityTest.cpp
        Stress_Tests/DeterminismTest.cpp
        Stress_Tests/MultibodyCrashTest.cpp
        Stress_Tests/ExtremeCcdTest.cpp
        Stress_Tests/LowEndBenchmark.cpp
        )

  foreach(src ${STRESS_TEST_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})

    if(MSVC)
      target_compile_options(${name} PRIVATE /W4
                $<$<CONFIG:Release>:/O2 /DNDEBUG>
                $<$<CONFIG:Debug>:/Od /Zi>
            )
    else()
      target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic
          $<$<CONFIG:Release>:-O3>
          $<$<CONFIG:Debug>:-g -O0>
      )
    endif()

    target_link_libraries(${name} PRIVATE ciphyxs)
    # Add the project root so stress tests can use "include/ciphyxs/CiPhyxs.hpp" relative includes.
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})

    # Per-test timeouts — with spatial hash cell size fix + island threshold.
    # Slowest: ChaosDensityTest (~60-90s with previous hash issue),
    # DeterminismTest (~10-20s), LowEndBenchmark (~20-50s).
    if(${name} STREQUAL "ChaosDensityTest")
      set_tests_properties(${name} PROPERTIES TIMEOUT 360)
    elseif(${name} STREQUAL "MultibodyCrashTest")
      set_tests_properties(${name} PROPERTIES TIMEOUT 30)
    elseif(${name} STREQUAL "LowEndBenchmark")
      set_tests_properties(${name} PROPERTIES TIMEOUT 180)
    elseif(${name} STREQUAL "DeterminismTest")
      set_tests_properties(${name} PROPERTIES TIMEOUT 60)
    elseif(${name} STREQUAL "DeepStackTest")
      set_tests_properties(${name} PROPERTIES TIMEOUT 120)
    else()
      set_tests_properties(${name} PROPERTIES TIMEOUT 30)
    endif()
  endforeach()

  add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60
        DEPENDS ${TEST_SOURCES})

  # Custom target to run stress tests.
  add_custom_target(stress_check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 300
        DEPENDS ${STRESS_TEST_SOURCES})
endif()

# ─── Install ─────────────────────────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/ciphyxs
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ciphyxs EXPORT CiPhyxsTargets)
install(EXPORT CiPhyxsTargets
        FILE CiPhyxsTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CiPhyxs)

# ─── Doxygen (opt-in) ───────────────────────────────────────────────────────────────────────────
find_package(Doxygen QUIET)
if(Doxygen_FOUND)
  # Only build docs when explicitly requested:  cmake --build . --target doc
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(DOXYGEN_GENERATE_TREEVIEW YES)

  configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
        "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
        @ONLY
    )

  doxygen_add_docs(doc
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ciphyxs"
        "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        COMMENT "Generating CiPhyxs API documentation"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
