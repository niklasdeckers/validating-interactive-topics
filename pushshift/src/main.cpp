#include <boost-iostreams/filter/zstd.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include <rapidjson/document.h>

#include <cmath>
#include <thread>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>


std::string_view subreddit{"captionthis"};

int main(int argc, char *argv[]) {
    const size_t numThreads = 16;

    std::filesystem::path base{"<point me at the submission corpurs and public comments corpus>"};

    std::vector<std::filesystem::path> files;
    for (const auto& path : std::filesystem::directory_iterator{base})
        if (path.is_regular_file() && path.path().extension() == ".zst")
            files.emplace_back(path.path());
    
    const size_t filesPerThread = std::ceil(files.size() / (float)numThreads);

    std::cout << "Processing " << files.size() << " files using " << numThreads << " threads" << " -- " << filesPerThread << " files per thread" << std::endl;

    std::vector<std::jthread> threads;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back([files, i, filesPerThread] {
            boost::iostreams::filtering_ostream out;
            out.push(boost::iostreams::gzip_compressor());
            out.push(boost::iostreams::file_sink(std::format("output-{}.jsonl.gz", i)));
            for (size_t j = i*filesPerThread; j < files.size() && j < (i+1)*filesPerThread; ++j) {
                std::cout << "Thread " << i << ": " << (j-(i*filesPerThread)+1) << "/" << filesPerThread << std::endl;
                const auto& path = files[j];
                std::ifstream file(path.c_str());
                assert(file.good());
                boost::iostreams::filtering_istream stream{boost_mod::iostreams::zstd_decompressor()};
                stream.push(file);
                for (std::string line; std::getline(stream, line);) {
                    if (line.find(subreddit) != std::string::npos) { // Contains the subreddit somewhere in the line as a heuristic for checking that it is the subreddit before parsing as json.
                        rapidjson::Document d;
                        d.Parse(line.c_str());
                        if (d["subreddit"].GetString() == subreddit)
                            out << line << '\n';
                    }
                }
                out << std::flush;
            }
            std::cout << "Thread " << i << ": Done" << std::endl;
            out.reset();
        });
    }
}
