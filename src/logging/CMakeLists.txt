add_library(mg-logging STATIC)
add_library(mg::logging ALIAS mg-logging)

target_sources(mg-logging
        PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ..
        FILES
        log.hpp
        init.hpp

        PRIVATE
        log.cpp
        init.cpp
)

find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)

target_link_libraries(mg-logging
        PUBLIC spdlog::spdlog fmt::fmt
        PRIVATE mg-utils gflags)
