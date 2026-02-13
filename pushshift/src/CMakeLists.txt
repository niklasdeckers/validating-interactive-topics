add_executable(filterreddit
    main.cpp
    boost-iostreams/zstd.cpp
)

target_include_directories(filterreddit PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(filterreddit PUBLIC cxx_std_20)

##########################################################################################
# Libraries
##########################################################################################
# zstd compression
CPMAddPackage(URI "gh:facebook/zstd@1.5.7" EXCLUDE_FROM_ALL YES SOURCE_SUBDIR "build/cmake")
target_link_libraries(filterreddit PRIVATE libzstd)

# boost::iostream for streamed decompression
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ENABLE_CMAKE ON)
find_package(Boost 1.45.0 COMPONENTS iostreams serialization REQUIRED)
target_link_libraries(filterreddit PUBLIC Boost::iostreams Boost::serialization)

# RapidJSOM
CPMAddPackage(URI "gh:Tencent/rapidjson@1.1.0" EXCLUDE_FROM_ALL YES)
# target_link_libraries(filterreddit PUBLIC RapidJSON)
target_include_directories(filterreddit PUBLIC ${rapidjson_SOURCE_DIR}/include)
