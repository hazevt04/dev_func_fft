cmake_minimum_required( VERSION 3.8.0 )

project( try_fft LANGUAGES CXX )

include_directories( "include" )

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/main.cpp"
   )

add_executable( "${EXEC_NAME}" "${SRCS}" )
target_link_libraries( "${EXEC_NAME}" m )

set_target_properties( ${EXEC_NAME}
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

