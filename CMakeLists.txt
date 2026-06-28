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
        test_alignment.cpp
        test_collision.cpp
        test_final.cpp
        test_joints.cpp
        test_new_features.cpp
        test_parallel.cpp
        test_sap.cpp
        test_convexmesh.cpp
        test_trimesh.cpp
        test_ccd.cpp
        test_ragdoll.cpp
        test_dbvt.cpp
        test_contactreduction.cpp
        test_vehicle.cpp
        test_character.cpp
        test_taskgraph.cpp
        test_softbody.cpp
        test_fracture.cpp
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
                $<$<CONFIG:Release>:-O2>
                $<$<CONFIG:Debug>:-g -O0>
            )
        endif()

        target_link_libraries(${name} PRIVATE ciphyxs)
        add_test(NAME ${name} COMMAND ${name})
    endforeach()

    # ─── Stress-Test Suite ──────────────────────────────────────────────────────────────────────
    set(STRESS_TEST_SOURCES
        Stress_Tests/DeepStackTest.cpp
        Stress_Tests/TunnelingTest.cpp
        Stress_Tests/CCDConvexMeshTest.cpp
        Stress_Tests/ChaosDensityTest.cpp
        Stress_Tests/DeterminismTest.cpp
        Stress_Tests/MultibodyCrashTest.cpp
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
                $<$<CONFIG:Release>:-O2>
                $<$<CONFIG:Debug>:-g -O0>
            )
        endif()

        target_link_libraries(${name} PRIVATE ciphyxs)
        add_test(NAME ${name} COMMAND ${name})
    endforeach()

    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${TEST_SOURCES})

    # Custom target to run stress tests.
    add_custom_target(stress_check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
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
