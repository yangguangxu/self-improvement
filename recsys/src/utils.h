#ifndef UTILS_H_H
#define UTILS_H_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>
#include <tbb/concurrent_unordered_map.h>
#include <thread>
#include <mutex>

void Split(const std::string& line, const std::string& sep, std::vector<std::string>* tokens, bool allow_empty_tok=false);

uint32_t MurmurHash(const std::string& key);

uint32_t MurmurHash(const char*key, int32_t len);

template <typename K, typename V>
class Accumulator: public tbb::concurrent_unordered_map<K, V> {
    std::vector<std::mutex> put_lock_;
public:
    Accumulator(int32_t lock_size=1024): put_lock_(lock_size) {}

    inline void PutWithLock(const K& key, const V& val) {
        auto hash = MurmurHash(reinterpret_cast<const char*>(&key), sizeof(K));
        auto lock_index = hash % put_lock_.size();
        std::lock_guard<std::mutex> lock(put_lock_[lock_index]);
        auto it = this->find(key);
        if (it == this->end()) {
            this->insert({key, val});
        } else {
            it->second += val;
        }
    }

    inline bool Get(const K& key, V* val) const {
        auto it = find(key);
        if (it != this->end()) {
            *val = it->second;
            return true;
        }
        return false;
    }
};

template<typename T, typename A>
bool HasCommon(
    const std::vector<T, A>& first,
    const std::vector<T, A>& second) {
    auto i = first.begin();
    auto j = second.begin();

    while (i != first.end() && j != second.end()) {
        if (*i < *j) {
            ++i;
        } else if (*i > *j) {
            ++j;
        } else {
            return true;
        }
    }

    return false;
}

#endif //UTILS_H_H
