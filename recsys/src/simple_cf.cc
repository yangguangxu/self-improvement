#include <omp.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <gflags/gflags.h>
#include "utils.h"

static const int32_t kClassicSimilarity = 0;
static const int32_t kPearsonSimilarity = 1;
static const int32_t kCosinSimilarity = 2;

DEFINE_bool(hot_penalty, false, "if turn on hot penalty");
DEFINE_int32(similarity_type, kClassicSimilarity,
          "0 for classic similarity, 1 for pearson correlation, 2 for cosin similarity");

namespace cf {

using user_id_t = uint32_t;
using item_id_t = uint32_t;
using user_id_pair = int64_t;

struct UserItemInfo {
    user_id_t user_id;
    item_id_t item_id;
    float weight;
};

struct SimilarUser {
    user_id_t u;
    user_id_t v;
    float simi;
};

struct InteractStats {
    float product = 0.0;
    uint32_t count = 0;
    float norm_1 = 0.0;
    float norm_2 = 0.0;

    InteractStats(float p=0, uint32_t c=0, float n1=0.0, float n2=0.0):
              product(p), count(c), norm_1(n1), norm_2(n2) {}

    void operator += (const InteractStats& that) {
        product += that.product;
        count += that.count;
        if (FLAGS_similarity_type == kPearsonSimilarity) {
            norm_1 += that.norm_1;
            norm_2 += that.norm_2;
        }
    }
};

template<class T>
class StrIdMap {
    std::unordered_map<std::string, T> id_map_;
    std::vector<std::string> str_map_;
    std::string empty_;
public:
    inline T GetOrAdd(const std::string& str) {
        auto it = id_map_.find(str);
        if (it != id_map_.end()) {
            return it->second;
        } else {
            auto id = str_map_.size();
            str_map_.push_back(str);
            id_map_.insert(std::make_pair(str, id));
            return id;
        }
    }

    inline const std::string& GetStr(const T& id) {
        if (0 <= id && id < str_map_.size()) {
            return str_map_[id];
        } else {
            return empty_;
        }
    }

    inline size_t Size() {
        return str_map_.size();
    }
};

std::ostream& operator<<(std::ostream& os, const std::vector<UserItemInfo>& l) {
    os << "---------------" << std::endl;
    for (auto& t : l) {
        os << t.user_id << " : " << t.item_id << " : " << t.weight << std::endl;
    }
    os << "---------------" << std::endl;
    return os;
}

class SimpleTrainer {
public:
    bool LoadDataset(std::istream& in) {
        if (!in.good()) {
            std::cerr << "Failed to load data set, broken stream" << std::endl;
            return false;
        }
        max_item_id = 0;
        max_user_id = 0;
        std::string line;
        std::vector<std::string> toks;
        while(getline(in, line)) {
            Split(line, " ", &toks);
            if (toks.size() != 3) {
                std::cerr<<"Skip illegal line: "<<line<<std::endl;
                continue;
            }
            auto user_id = user_id_map.GetOrAdd(toks[0]);
            auto item_id = item_id_map.GetOrAdd(toks[1]);
            auto weight = stof(toks[2]);
            data.push_back({user_id, item_id, weight});
            max_item_id = std::max(item_id, max_item_id);
            max_user_id = std::max(user_id, max_user_id);
        }
        std::cerr << "users: " << user_id_map.Size() << std::endl;
        std::cerr << "items: " << item_id_map.Size() << std::endl;
        return true;
    }

    void Run() {
        auto tick_0 = std::chrono::high_resolution_clock::now();
        BuildIndex();
        auto tick_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> index_timer = tick_1 - tick_0;
        std::cerr << "BuildIndex take: "
                  << index_timer.count() / 1000.0 << " seconds" << std::endl;
        ComputeHotPenalty();
        auto tick_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> penalty_timer = tick_2 - tick_1;
        std::cerr << "ComputeHotPenalty take: "
                  << penalty_timer.count() / 1000.0 << " seconds" << std::endl;
        CentralizeWeightByUser();
        auto tick_3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> centralize_timer = tick_3 - tick_2;
        std::cerr << "CentralizeWeightByUser take: "
                  << centralize_timer.count() / 1000.0 << " seconds" << std::endl;
        ComputeUserInteractWeight();
        auto tick_4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> interact_timer = tick_4 - tick_3;
        std::cerr << "ComputeUserInteractWeight take : "
                  << interact_timer.count() / 1000.0 << " seconds" << std::endl;
        ComputeUserVectorNorm();
        auto tick_5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> vector_norm_timer = tick_5 - tick_4;
        std::cerr << "ComputeUserVectorNorm take "
                  << vector_norm_timer.count() / 1000.0 << " seconds" << std::endl;
        GenerateTable();
        auto tick_6 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gen_table_timer = tick_6 - tick_5;
        std::cerr << "GenerateTable take: "
                  << gen_table_timer.count() / 1000.0 << " seconds" << std::endl;
    }

    void CentralizeWeightByUser() {
        // For pearson correlation
        if (FLAGS_similarity_type != kPearsonSimilarity) return;

        std::vector<float> user_average(max_user_id + 1, 0.0);
        std::vector<uint32_t> user_count(max_user_id + 1, 0);
        for (const auto& t : data) {
            user_average[t.user_id] += t.weight;
            user_count[t.user_id] += 1;
        }
        for (user_id_t user_id = 0; user_id <= max_user_id; ++user_id) {
            if (user_count[user_id]) {
                user_average[user_id] /= user_count[user_id];
            }
        }
        for (auto& t : data) {
            t.weight -= user_average[t.user_id];
        }
    }

    inline void BeforeIterate() {
        read_offset = 0;
    }

    bool Iterate(std::string* user, std::vector< std::pair<std::string, float> >* neighbors, int top_n=50)  {
        assert(top_n >= 0);
        neighbors->clear();

        auto size = simi_users.size();
        if (read_offset >= size) return false;
        if (top_n == 0) return true;

        //make sure read_offset is head of list
        assert(!read_offset || simi_users[read_offset - 1].u != simi_users[read_offset].u);

        auto current_u = simi_users[read_offset].u;
        *user = user_id_map.GetStr(current_u);
        for (int i = 0; read_offset < size; ++read_offset, ++i) {
            if (simi_users[read_offset].u != current_u) break;
            auto& info = simi_users[read_offset];
            if (i < top_n) {
                auto v = user_id_map.GetStr(info.v);
                neighbors->push_back(std::make_pair(v, info.simi));
            }
        }
        return true;
    }

private:
    void BuildIndex() {
        item_index.clear();
        item_index.resize(max_item_id + 2, 0);
        sort(data.begin(), data.end(), group_same_item);
        item_index[data[0].item_id] = 0;
        for (uint32_t i = 0; i < data.size(); ++i) {
            if (i + 1 < data.size() && data[i].item_id != data[i + 1].item_id) {
                item_index[data[i + 1].item_id] = i + 1;
            }
        }
        item_index[max_item_id + 1] = data.size();
    }

    void ComputeUserVectorNorm() {
        if (FLAGS_similarity_type == kPearsonSimilarity) return;

        user_vector_norm.clear();
        user_vector_norm.resize(max_user_id + 1, 0.0);

        if (FLAGS_similarity_type == kCosinSimilarity) {
            for (const auto& t : data) {
                user_vector_norm[t.user_id] += t.weight * t.weight;
            }
        } else {
            for (const auto& t : data) {
                user_vector_norm[t.user_id] += 1;
            }
        }

        # pragma omp parallel for schedule(static, 1024)
        for (uint32_t i = 0; i <= max_user_id; ++i) {
            user_vector_norm[i] = sqrt(user_vector_norm[i]);
        }
    }

    void ComputeUserInteractWeight() {
        omp_set_nested(1);
        std::atomic<int32_t> counter(0);
        # pragma omp parallel for
        for (item_id_t item_id = 0; item_id <= max_item_id; ++item_id) {
            auto b = item_index[item_id];
            auto e = item_index[item_id + 1];
            # pragma omp parallel for
            for (uint32_t i = b; i < e; ++i) {
                for (uint32_t j = i + 1; j < e; ++j) {
                    auto u = static_cast<uint64_t>(data[i].user_id);
                    auto v = static_cast<uint64_t>(data[j].user_id);

                    float product = (FLAGS_similarity_type == kClassicSimilarity ? 1.0 : data[i].weight * data[j].weight);
                    product /= item_penalty[item_id];

                    uint64_t key = 0;
                    if (u < v) key = (u << 32) | v;
                    else {
                        key = (v << 32) | u;
                    }

                    InteractStats stats(product, 1);
                    // for pearson correlation
                    if (FLAGS_similarity_type == kPearsonSimilarity) {
                        float n1 = data[i].weight * data[i].weight;
                        float n2 = data[j].weight * data[j].weight;
                        if (u < v) {
                            stats.norm_1 = n1;
                            stats.norm_2 = n2;
                        } else {
                            stats.norm_1 = n2;
                            stats.norm_2 = n1;
                        }
                    }

                    user_interact_stats.PutWithLock(key, stats);
                }
            }
            counter += 1;
            if (counter % 10000 == 0) {
                std::cerr << "interact weight complete " << counter.load()
                          << " / " << max_item_id + 1 << std::endl;
            }
        }
    }

    void ComputeHotPenalty() {
        item_penalty.clear();
        item_penalty.resize(max_item_id + 1, 1.0);
        if (!FLAGS_hot_penalty) return;
        # pragma omp parallel for schedule(static, 1024)
        for (uint32_t i = 0; i <= max_item_id; ++i) {
            auto b = item_index[i];
            auto e = item_index[i + 1];
            item_penalty[i] = log(1.0 + e - b);
        }
    }

    void GenerateTable() {
        for (auto it = user_interact_stats.begin();
                  it != user_interact_stats.end(); ++it) {
            auto u = static_cast<user_id_t>((it->first >> 32) & 0xffffffff);
            auto v = static_cast<user_id_t>(it->first & 0xffffffff);
            auto count = it->second.count;
            float u_norm, v_norm, weight;

            if (FLAGS_similarity_type == kPearsonSimilarity) {
                u_norm = sqrt(it->second.norm_1);
                v_norm = sqrt(it->second.norm_2);
                weight = it->second.product / (u_norm * v_norm);
                weight = std::min(count / 50.0, 1.0) * weight;
            } else {
                u_norm = user_vector_norm[u];
                v_norm = user_vector_norm[v];
                weight = it->second.product / (u_norm * v_norm);
            }

            simi_users.push_back({u, v, weight});
            simi_users.push_back({v, u, weight});
        }
        std::sort(simi_users.begin(), simi_users.end(), group_same_user);

        std::cerr<<"GenTable succeed."<<std::endl;
    }

    static bool group_same_item(const UserItemInfo& a, const UserItemInfo& b) {
        if (a.item_id != b.item_id) return a.item_id < b.item_id;
        if (a.user_id != b.user_id) return a.user_id < b.user_id;
        return a.weight > b.weight;
    }

    static bool group_same_user(const SimilarUser& a, const SimilarUser& b) {
        if (a.u != b.u) return a.u < b.u;
        return a.simi > b.simi;
    }

    StrIdMap<user_id_t> user_id_map;
    StrIdMap<item_id_t> item_id_map;
    std::vector<uint32_t> item_index;
    std::vector<float> item_penalty;
    std::vector<UserItemInfo> data;
    std::vector<float> user_vector_norm;
    Accumulator<uint64_t, InteractStats> user_interact_stats;
    std::vector<SimilarUser> simi_users;
    size_t read_offset;
    uint32_t max_item_id;
    uint32_t max_user_id;
};

}

DEFINE_string(input, "", "input file path");
DEFINE_string(output, "", "output file path");
DEFINE_int32(topn, 50, "topn neighbors");
DEFINE_double(threshold, 0.0, "filter neighbors with similarity less than threshold");

int main(int argc, char** argv) {
    auto begin_tick = std::chrono::high_resolution_clock::now();
    bool status = false;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::ofstream os(FLAGS_output.c_str());
    if (!os.good()) {
        std::cerr << "Can't open file " << FLAGS_output
                  << " for output " << std::endl;
        return -1;
    }

    std::ifstream in(FLAGS_input.c_str());
    if (!in.good()) {
        std::cerr << "Can't open file " << FLAGS_input
                  << " for input " << std::endl;
        return -1;
    }

    cf::SimpleTrainer trainer;
    status = trainer.LoadDataset(in);
    if (!status) return -1;

    trainer.Run();

    std::string user_id;
    std::vector<std::pair<std::string, float> > similar_user_list;
    trainer.BeforeIterate();
    while (trainer.Iterate(&user_id, &similar_user_list, FLAGS_topn)) {
        os << user_id << "\01";
        bool first = true;
        for (const auto& t : similar_user_list) {
            if (t.second < FLAGS_threshold) continue;
            if (!first) os << "\01";
            first = false;
            os << t.first << "\02" << t.second;
        }
        os << std::endl;
    }
    os.close();
    auto end_tick = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = end_tick - begin_tick;
    std::cerr << "Total Time used: "
              << total_time.count() / 60000.0 << " seconds" << std::endl;
    return 0;
}
