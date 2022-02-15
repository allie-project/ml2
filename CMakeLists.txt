cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0042 NEW)
set(CMAKE_CXX_STANDARD 14)

project(glide)

file(GLOB_RECURSE SRCS "src/**/*.cc")
add_library(${PROJECT_NAME} SHARED ${SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtl/include
)
set(XTENSOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(NOT GLIDE_NO_OPENMP)
	find_package(OpenMP REQUIRED)
	if (OPENMP_FOUND)
		find_package(Threads REQUIRED)
		add_library(OpenMP::OpenMP_CXX_xtensor IMPORTED INTERFACE)
		set_property(TARGET OpenMP::OpenMP_CXX_xtensor PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
		set_property(TARGET OpenMP::OpenMP_CXX_xtensor PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_LIBRARIES} Threads::Threads)

		message(STATUS "OpenMP found!")
	else()
		message(WARNING "OpenMP not found!")
	endif()
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC XSIMD_ENABLE_FALLBACK=1 XSIMD_ENABLE_XTL_COMPLEX=1)

target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${XTENSOR_INCLUDE_DIR} src> $<INSTALL_INTERFACE:src>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_definitions(-DXTENSOR_ENABLE_ASSERT)
	add_definitions(-DXTENSOR_ENABLE_CHECK_DIMENSION)
endif()

add_definitions(-DXTENSOR_DEFAULT_LAYOUT=layout_type::column_major)

target_compile_definitions(${PROJECT_NAME} PUBLIC
	XTENSOR_USE_XSIMD
	XTENSOR_DISABLE_EXCEPTIONS
	XTL_DISABLE_EXCEPTIONS
)
if(NOT GLIDE_NO_OPENMP)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX_xtensor)
endif()

if(MSVC)
	add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>)
	target_compile_definitions(${PROJECT_NAME} PRIVATE _SCL_SECURE_NO_WARNINGS)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/nologo
		/EHsc
		/arch:SSE /arch:SSE2
		/arch:AVX /arch:AVX2
		/wd4396 /wd4309 /wd4244
	)
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		target_compile_options(${PROJECT_NAME} PRIVATE
			/ltcg /MP /bigobj
			/Zi
			/Gy /GF /GL
			/DNDEBUG
			/OPT:REF /OPT:ICF
		)
	endif()
endif()

string(
	APPEND _opts
	"$<IF:$<CXX_COMPILER_ID:MSVC>,"
		"/W2;$<$<CONFIG:RELEASE>:/O2>,"
		"-W2;$<$<CONFIG:RELEASE>:-O3>"
	">"
)
target_compile_options(${PROJECT_NAME} PUBLIC ${_opts})

enable_testing()

file(GLOB_RECURSE TEST_SRCS "test/**/*.test.cc")
add_executable(${PROJECT_NAME}-test ${SRCS} ${TEST_SRCS})
target_include_directories(${PROJECT_NAME}-test PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtl/include
)
target_compile_definitions(${PROJECT_NAME}-test PUBLIC
	XTENSOR_USE_XSIMD
	XTENSOR_DISABLE_EXCEPTIONS
	XTL_DISABLE_EXCEPTIONS
)
target_link_libraries(${PROJECT_NAME}-test gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-test)