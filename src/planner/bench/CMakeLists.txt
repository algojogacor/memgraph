add_executable(
    memgraph__benchmark__new_planner
)

target_sources(memgraph__benchmark__new_planner
    PRIVATE
    bench_main.cpp
    bench_union_find.cpp
    bench_egraph_rebuild.cpp
    bench_pattern_vm.cpp
    bench_vm.cpp
    bench_join.cpp
    bench_rewrite.cpp
    bench_extract.cpp
)

target_include_directories(memgraph__benchmark__new_planner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(memgraph__benchmark__new_planner PRIVATE mg::planner mg::planner::test_support benchmark::benchmark)

add_dependencies(memgraph__benchmark memgraph__benchmark__new_planner)
