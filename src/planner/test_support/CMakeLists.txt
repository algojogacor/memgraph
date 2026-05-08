add_library(mg-planner-test-support OBJECT)
add_library(mg::planner::test_support ALIAS mg-planner-test-support)

target_sources(mg-planner-test-support
    PRIVATE
    instantiations.cpp
)

target_include_directories(mg-planner-test-support
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(mg-planner-test-support PUBLIC mg::planner)
