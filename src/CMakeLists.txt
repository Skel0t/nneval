set(ART_SRCS
    nn.art
    image.art
    matrix.art
    interface.art
    output.art
    iterate.art
    kernel.art
)

set(NN_SRCS
    nn.h
    nn.cpp
    interface.h
    image.h
    file_path.h
    test.cpp
    image.cpp
    interface.cpp
)

anydsl_runtime_wrap(NN_OBJS
    NAME "nn"
    FRONTEND artic
    CLANG_FLAGS ${CLANG_FLAGS}
    ARTIC_FLAGS --log-level info
    FILES ${ART_SRCS})

find_package(PNG REQUIRED)
find_package(LZ4 REQUIRED)

add_executable(nneval ${NN_SRCS} ${NN_OBJS})
target_include_directories(nneval PUBLIC ${LZ4_INCLUDE_DIR})
target_link_libraries(nneval ${AnyDSL_runtime_LIBRARIES} ${LZ4_LIBRARY} ${PNG_LIBRARIES})

if (COLORIZE)
    target_compile_definitions(nneval PUBLIC -DCOLORIZE)
endif()