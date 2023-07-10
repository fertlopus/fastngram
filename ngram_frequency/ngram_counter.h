#ifndef NGRAM_COUNTER_H
#define NGRAM_COUNTER_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <locale>
#include <codecvt>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <thread>
#include <mutex>

class NgramCounter
{
private:
    const std::wstring corpora;
    const int64_t max_ngram_size;
    const double threshold;
    const double epsilon;
    const int64_t n_jobs;
    int64_t corpora_length;
    int64_t bucket_size;
    int64_t occurrence_lower_bound;
    std::vector<int64_t> ngram_size_list;
    std::vector<std::pair<std::wstring, int64_t>> counted_data;
    std::mutex mtx;

public:
    NgramCounter(const std::wstring& _corpora,
                 const int64_t _max_ngram_size,
                 const double _threshold,
                 const double _epsilon,
                 const int64_t _n_jobs);
    ~NgramCounter();
    void count_ngram();
    void count_ngram_each(const int64_t ngram_size);
    void extract_all_ngram_to_csv(const std::string ngram_count_path);
    void extract_all_ngram(std::unordered_map<std::wstring, int64_t>& placeholder);
    void extract_top_ngram_to_csv(const std::string ngram_count_top_path, const int64_t extract_num);
    void extract_top_ngram(std::vector<std::wstring>& vocabulary, std::vector<int64_t>& count_vocabulary, const int64_t extract_num);
};
#endif