cmake_minimum_required(VERSION 3.12)

# Set the project name
project(tinyode VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force coloured compiler output
add_compile_options(-fdiagnostics-color)

# CXX flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find required packages
find_package(Catch2 2.13 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Libraries
set(LIBS Eigen3::Eigen)

# Target names
set(TARGET_LIB tinyodeLib)
set(TARGET_BIN tinyodeExecutable)
set(TARGET_TEST tinyodeTests)

# Source files
file(GLOB SRC_INCLUDES "include/*.hpp")
file(GLOB SRC_EXAMPLES "examples/*.cpp")
file(GLOB SRC_TEST "test/*.cpp")

# Static library
if(SRC_INCLUDES)
  add_library(${TARGET_LIB} STATIC ${SRC_INCLUDES})
  target_link_libraries(${TARGET_LIB} ${LIBS})
endif()
set_target_properties(${TARGET_LIB} PROPERTIES LINKER_LANGUAGE CXX)

# Examples
foreach(SRC_FILE ${SRC_EXAMPLES})
  get_filename_component(EXAMPLE ${SRC_FILE} NAME_WE)
  add_executable(${EXAMPLE} ${SRC_FILE})
  if(SRC_INCLUDES)
    target_link_libraries(${EXAMPLE} ${TARGET_LIB})
  endif()
  target_link_libraries(${EXAMPLE} ${LIBS})
endforeach()

# Unit tests
add_executable(${TARGET_TEST} ${SRC_TEST})
if(SRC_INCLUDES)
  target_link_libraries(${TARGET_TEST} ${TARGET_LIB})
endif()
target_link_libraries(${TARGET_TEST} ${LIBS})
target_link_libraries(${TARGET_TEST} Catch2::Catch2)

# Run unit tests after building executables
add_custom_target(
  run_tests ALL
  COMMAND ${TARGET_TEST} --use-colour yes
  DEPENDS ${TARGET_TEST} ${TARGET_LIB}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running tests")

# Create a library target with the include files
add_library(tinyode SHARED ${SRC_INCLUDES})

# Link libraries to the library target
target_link_libraries(tinyode PUBLIC ${LIBS})
set_target_properties(tinyode PROPERTIES LINKER_LANGUAGE CXX)

# Set the installation rules
install(
  TARGETS tinyode
  EXPORT tinyodeTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include/tinyode)

install(DIRECTORY include/ DESTINATION include/tinyode)

install(
  EXPORT tinyodeTargets
  FILE tinyodeTargets.cmake
  NAMESPACE tinyode::
  DESTINATION lib/cmake/tinyode)

# Create a package configuration file for library
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "tinyodeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES "tinyodeConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/tinyodeConfigVersion.cmake"
        DESTINATION lib/cmake/tinyode)
