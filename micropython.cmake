add_library(uumpy INTERFACE)

target_sources(uumpy INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/linalg.c
        ${CMAKE_CURRENT_LIST_DIR}/moduumpy.c
        ${CMAKE_CURRENT_LIST_DIR}/reductions.c
        ${CMAKE_CURRENT_LIST_DIR}/ufunc.c
        ${CMAKE_CURRENT_LIST_DIR}/uumath.c
)

target_include_directories(uumpy INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE uumpy)
