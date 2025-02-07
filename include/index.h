#ifndef STKQ_INDEX_H
#define STKQ_INDEX_H

#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <functional>
#include <map>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "distance.h"
#include "parameters.h"
#include "policy.h"
#include "rtree.h"
#include "CommonDataStructure.h"
#include <mm_malloc.h>
#include <stdlib.h>
#define INF_N -std::numeric_limits<float>::max()
#define INF_P std::numeric_limits<float>::max()

namespace stkq
{
    typedef std::lock_guard<std::mutex> LockGuard;
    class NNDescent
    {
    public:
        unsigned K;
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;
        struct Neighbor
        {
            unsigned id;
            float distance;
            bool flag;

            Neighbor() = default;

            Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

            inline bool operator<(const Neighbor &other) const
            {
                return distance < other.distance;
            }

            inline bool operator>(const Neighbor &other) const
            {
                return distance > other.distance;
            }
        };

        struct nhood
        {
            std::mutex lock;            // 互斥锁 用于并发访问控制
            std::vector<Neighbor> pool; // 存储邻居节点的优先队列 最小堆
            unsigned M;                 // 记录pool的大小

            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;

            nhood() {}

            nhood(unsigned l, unsigned s)
            {
                M = s;
                nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                // resize用于定义向量的实际大小（元素数量） 而reserve用于优化内存分配以应对向量大小的潜在增长
                pool.reserve(l);
            }

            nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N)
            {
                M = s;
                nn_new.resize(s * 2);
                GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N); // 还用随机数填充 nn_new
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(const nhood &other)
            {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void insert(unsigned id, float dist)
            {
                // 方法用于向 pool 中插入新的邻居节点，如果新节点的距离小于 pool 中最远节点的距离，则替换之
                //  pool 最大堆
                LockGuard guard(lock);
                if (dist > pool.front().distance)
                    return;
                for (unsigned i = 0; i < pool.size(); i++)
                {
                    if (id == pool[i].id)
                        return;
                }
                if (pool.size() < pool.capacity())
                {
                    pool.push_back(Neighbor(id, dist, true));
                    std::push_heap(pool.begin(), pool.end());
                }
                else
                {
                    std::pop_heap(pool.begin(), pool.end());
                    // 把最大的放到最后一个
                    pool[pool.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(pool.begin(), pool.end());
                }
            }

            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    // 遍历 nn_new（新邻居）集合中的每个元素 i
                    for (unsigned const j : nn_new)
                    {
                        // 再次遍历 nn_new 集合中的每个元素 j
                        if (i < j)
                        {
                            callback(i, j);
                            // 如果 i 小于 j（避免重复处理和自连接），调用 callback 函数处理这对邻居节点
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        callback(i, j);
                        // 接着遍历 nn_old（旧邻居）集合中的每个元素 j，对每个新旧邻居对调用 callback 函数
                    }
                }
            }
            // 似乎是NN Descent
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
        {
            // find the location to insert
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance)
            {
                memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance)
            {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1)
            {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)
                    right = mid;
                else
                    left = mid;
            }
            // check equal ID

            while (left > 0)
            {
                if (addr[left].distance < nn.distance)
                    break;
                if (addr[left].id == nn.id)
                    return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)
                return K + 1;
            memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        typedef std::vector<nhood> KNNGraph;
        KNNGraph graph_;
    };

    class NSW
    {
    public:
        unsigned NN_;
        unsigned ef_construction_ = 150; // l
        unsigned n_threads_ = 32;
    };

    class HNSW
    {
    public:
        unsigned m_ = 12; // k
        unsigned max_m_ = 12;
        unsigned max_m0_ = 24;
        int mult;
        float level_mult_ = 1 / log(1.0 * m_);

        int max_level_ = 0;
        mutable std::mutex max_level_guard_;

        template <typename KeyType, typename DataType>
        class MinHeap
        { // 这是一个实现最小堆数据结构的类。最小堆是一种树形数据结构，用于高效地管理和检索元素，其中父节点的键（key）总是小于或等于其子节点的键
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key > i2.key;
                }
            };

            MinHeap()
            {
            }

            const KeyType top_key()
            {
                if (v_.size() <= 0)
                    return 0.0;
                return v_[0].key;
            }

            Item top()
            {
                if (v_.size() <= 0)
                    throw std::runtime_error("[Error] Called top() operation with empty heap");
                return v_[0];
            }
            // 返回堆顶元素 最小元素

            void pop()
            {
                std::pop_heap(v_.begin(), v_.end());
                v_.pop_back();
            }
            // 移除堆顶元素

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));
                std::push_heap(v_.begin(), v_.end());
            }

            // 向堆中添加新元素

            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class HnswNode
        {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level + 1)
            {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) { level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode *> &GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<HnswNode *> &new_friends)
            {
                if (level >= friends_at_layer_.size())
                    friends_at_layer_.resize(level + 1);
                friends_at_layer_[level].swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

            // 1. The list of friends is sorted
            // 2. bCheckForDup == true addFriend checks for duplicates using binary searching
            inline void AddFriends(HnswNode *element, bool bCheckForDup)
            {
                std::unique_lock<std::mutex> lock(access_guard_);
                if (bCheckForDup)
                {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if (it == friends_at_layer_[0].end() || (*it) != element)
                    {
                        friends_at_layer_[0].insert(it, element);
                    }
                }
                else
                {
                    friends_at_layer_[0].push_back(element);
                }
            }

        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;

            std::vector<std::vector<HnswNode *>> friends_at_layer_;
            std::mutex access_guard_;
        };

        class FurtherFirst
        {
        public:
            FurtherFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const FurtherFirst &n) const
            {
                return (distance_ < n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较低” 因此 FurtherFirst 实际上是用于构建最大堆的
        };

        class CloserFirst
        {
        public:
            CloserFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const CloserFirst &n) const
            {
                return (distance_ > n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        };

        typedef typename std::pair<HnswNode *, float> IdDistancePair;
        struct IdDistancePairMinHeapComparer
        {
            bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
            {
                return p1.second > p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;
        // IdDistancePairMinHeap 是一个 4-叉最小堆，使用 boost::heap::d_ary_heap 实现，并通过 IdDistancePairMinHeapComparer 来比较元素
        // 这样构造的堆会将最小的距离元素保持在顶部
        HnswNode *enterpoint_ = nullptr;
        std::vector<HnswNode *> nodes_;
    };

    class NSG
    {
    public:
        unsigned R_refine;
        unsigned L_refine;
        unsigned C_refine;

        unsigned ep_;
        unsigned width;
    };

    class SSG
    {
    public:
        float A;
        unsigned n_try;
        // unsigned width;

        std::vector<unsigned> eps_;
        unsigned test_min = INT_MAX;
        unsigned test_max = 0;
        long long test_sum = 0;
    };

    class baseline4
    {
    public:
        class BS4Node
        {
        public:
            explicit BS4Node(int id, int level, size_t max_m, size_t max_m0)
                : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level + 1)
            {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            inline size_t GetId() const { return (size_t)id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) { level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<BS4Node *> &GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<BS4Node *> &new_friends)
            {
                if (level >= friends_at_layer_.size())
                    friends_at_layer_.resize(level + 1);
                friends_at_layer_[level].swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

            // 1. The list of friends is sorted
            // 2. bCheckForDup == true addFriend checks for duplicates using binary searching
            inline void AddFriends(BS4Node *element, bool bCheckForDup)
            {
                std::unique_lock<std::mutex> lock(access_guard_);
                if (bCheckForDup)
                {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if (it == friends_at_layer_[0].end() || (*it) != element)
                    {
                        friends_at_layer_[0].insert(it, element);
                    }
                }
                else
                {
                    friends_at_layer_[0].push_back(element);
                }
            }

        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;
            std::vector<std::vector<BS4Node *>> friends_at_layer_;
            std::mutex access_guard_;
        };

        class BS4FurtherFirst
        {
        public:
            BS4FurtherFirst(BS4Node *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline BS4Node *GetNode() const { return node_; }
            bool operator<(const BS4FurtherFirst &n) const
            {
                return (distance_ < n.GetDistance());
            }

        private:
            BS4Node *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较低” 因此 FurtherFirst 实际上是用于构建最大堆的
        };

        class BS4CloserFirst
        {
        public:
            BS4CloserFirst(BS4Node *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline BS4Node *GetNode() const { return node_; }
            bool operator<(const BS4CloserFirst &n) const
            {
                return (distance_ > n.GetDistance());
            }

        private:
            BS4Node *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        };

        // typedef typename std::pair<BS4Node *, float> IdDistancePair;
        // struct IdDistancePairMinHeapComparer
        // {
        //     bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
        //     {
        //         return p1.second > p2.second;
        //     }
        // };
        // typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;
        // // IdDistancePairMinHeap 是一个 4-叉最小堆，使用 boost::heap::d_ary_heap 实现，并通过 IdDistancePairMinHeapComparer 来比较元素
        // // 这样构造的堆会将最小的距离元素保持在顶部
        std::vector<BS4Node *> baseline4_enterpoint_; // = nullptr;
        std::vector<std::vector<BS4Node *>> baseline4_nodes_;
        std::vector<int> baseline4_max_level_;
        mutable std::mutex bs4_max_level_guard_;
    };

    class DEG
    {
    public:
        struct DEGNeighbor
        {
            unsigned id_;
            float emb_distance_;
            float geo_distance_;
            unsigned layer_;
            std::vector<std::pair<float, float>> available_range;

            DEGNeighbor() = default;
            DEGNeighbor(unsigned id, float emb_distance, float geo_distance) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance)
            {
                available_range.emplace_back(0, 1);
            }
            DEGNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range) {}
            DEGNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range, unsigned l) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range), layer_(l) {}

            inline bool operator<(const DEGNeighbor &other) const
            {
                // return geo_distance_ < other.geo_distance_;
                // 较小的 geo_distance_ 值会被排序到较前的位置
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
            }
        };

        struct DEGSimpleNeighbor
        {
            unsigned id_;
            // unsigned layer_;
            // std::vector<std::pair<float, float>> available_range;
            std::vector<std::pair<int8_t, int8_t>> active_range;

            DEGSimpleNeighbor() = default;
            // DEGSimpleNeighbor(unsigned id, std::vector<std::pair<float, float>> range) : id_{id}, available_range(range) {}
            DEGSimpleNeighbor(unsigned id, std::vector<std::pair<int8_t, int8_t>> range) : id_{id}, active_range(range) {}
        };

        struct DEGNNDescentNeighbor
        {
            unsigned id_;
            float emb_distance_;
            float geo_distance_;
            bool flag;
            int layer_;
            DEGNNDescentNeighbor() = default;
            DEGNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer)
            {
            }
            inline bool operator<(const DEGNNDescentNeighbor &other) const
            {
                // return geo_distance_ < other.geo_distance_;
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
                // 较小的 geo_distance_ 值会被排序到较前的位置
            }
        };

        class DEGNode
        {
        public:
            explicit DEGNode(int id, int max_m)
                : id_(id), max_m_(max_m)
            {
                // friends.reserve(max_m_ + 1);
                // friends_for_search.reserve(max_m_ + 1);
                friends.clear();
                friends_for_search.clear();
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetMaxM() const { return max_m_; }
            // inline int GetLevel() const { return level_; }
            // inline void SetLevel(int level) { level_ = level; }
            inline void SetMaxM(int max_m) { max_m_ = max_m; }
            inline std::vector<DEGNeighbor> &GetFriends() { return friends; }

            inline void SetFriends(std::vector<DEGNeighbor> &new_friends)
            {
                friends.swap(new_friends);
            }

            inline std::vector<DEGSimpleNeighbor> &GetSearchFriends() { return friends_for_search; }

            inline void SetSearchFriends(std::vector<DEGSimpleNeighbor> &new_friends)
            {
                friends_for_search.swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

        private:
            int id_;
            // int level_;
            size_t max_m_;
            std::vector<DEGNeighbor> friends;
            std::vector<DEGSimpleNeighbor> friends_for_search;
            std::mutex access_guard_;
        };

        struct skyline_descent
        {
            std::mutex lock; // 互斥锁 用于并发访问控制
            std::vector<DEGNNDescentNeighbor> pool;
            std::vector<DEGNNDescentNeighbor> outlier; // 存储邻居节点的优先队列 最小堆
            unsigned M;                                // 记录pool的最大大小, 也就是candidate边数
            unsigned Q;                                // 记录pool在update的过程中candidate set需要考虑的layer 也即quality result
            unsigned num_layer;
            unsigned use_range;
            // unsigned *pstart;
            std::unordered_set<unsigned> Visited_Set;
            // unsigned pool_size;
            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;
            skyline_descent() {}

            skyline_descent(unsigned l, unsigned s, unsigned q)
            {
                M = l;
                Q = q;
                nn_new.reserve(s * 2);
                pool.reserve(4 * l * l);
            }

            skyline_descent(const skyline_descent &other)
            {
                M = other.M;
                Q = other.Q;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void findSkyline(std::vector<DEGNNDescentNeighbor> &points, std::vector<DEGNNDescentNeighbor> &skyline, std::vector<DEGNNDescentNeighbor> &remain_points)
            {
                // Sort points by x-coordinate
                // Sweep to find skyline
                float max_emb_dis = std::numeric_limits<float>::max();
                for (const auto &point : points)
                {
                    if (point.emb_distance_ < max_emb_dis)
                    {
                        skyline.push_back(point);
                        max_emb_dis = point.emb_distance_;
                    }
                    else
                    {
                        remain_points.emplace_back(point);
                    }
                }
                // O(n)
            }

            void init_neighor(std::vector<DEGNNDescentNeighbor> &insert_points)
            {
                LockGuard guard(lock);
                std::vector<DEGNNDescentNeighbor> skyline_result;
                std::vector<DEGNNDescentNeighbor> remain_points;
                int l = 0;
                while (!insert_points.empty())
                {
                    findSkyline(insert_points, skyline_result, remain_points);
                    insert_points.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, true, l);
                    }
                    outlier.swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(remain_points);
                    l++;
                    // if (l >= Q)
                    // {
                    //     break;
                    // }
                }
                num_layer = l;
                sort(pool.begin(), pool.end());
                std::vector<unsigned>().swap(nn_new);
                std::vector<unsigned>().swap(nn_old);
                std::vector<unsigned>().swap(rnn_new);
                std::vector<unsigned>().swap(rnn_old);
            }

            void updateNeighbor()
            {
                LockGuard guard(lock);
                std::vector<DEGNNDescentNeighbor> skyline_result;
                std::vector<DEGNNDescentNeighbor> remain_points;
                std::vector<DEGNNDescentNeighbor> candidate;
                candidate.clear();
                candidate.swap(pool);
                int l = 0;
                sort(candidate.begin(), candidate.end());
                while (pool.size() < M && candidate.size() > 0)
                {
                    findSkyline(candidate, skyline_result, remain_points);
                    candidate.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.flag, l);
                    }
                    outlier.swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(remain_points);
                    l++;
                    // if (l >= Q)
                    // {
                    //     break;
                    // }
                }
                num_layer = l;
            }

            void insert(unsigned id, float e_dist, float s_dist)
            {
                LockGuard guard(lock);
                // the simplest not very slow
                if (Visited_Set.find(id) != Visited_Set.end())
                {
                    return;
                }
                Visited_Set.insert(id);
                for (int i = 0; i < outlier.size(); i++)
                {
                    if (outlier[i].emb_distance_ <= e_dist && outlier[i].geo_distance_ <= s_dist)
                    {
                        return;
                    }
                    else if (outlier[i].geo_distance_ > s_dist)
                    {
                        break;
                    }
                }
                pool.emplace_back(id, e_dist, s_dist, true, -1);
                return;
            }

            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    // 遍历 nn_new（新邻居）集合中的每个元素 i
                    for (unsigned const j : nn_new)
                    {
                        // 再次遍历 nn_new 集合中的每个元素 j
                        if (i < j)
                        {
                            callback(i, j);
                            // 如果 i 小于 j（避免重复处理和自连接），调用 callback 函数处理这对邻居节点
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        // callback(i, j, insert_delete_time, search_time, compute_time_complexity, mtx);
                        // 接着遍历 nn_old（旧邻居）集合中的每个元素 j，对每个新旧邻居对调用 callback 函数
                        callback(i, j);
                    }
                }
            }
            // 似乎是NN Descent
        };

        struct skyline_queue
        {
            std::vector<DEGNNDescentNeighbor> pool;
            unsigned M; // 记录pool的最大大小, 也就是candidate边数
            unsigned num_layer;
            skyline_queue() {}

            skyline_queue(unsigned l)
            {
                M = l;
                pool.reserve(M);
            }

            skyline_queue(const skyline_queue &other)
            {
                M = other.M;
                pool.reserve(other.pool.capacity());
            }

            float cross(const DEGNNDescentNeighbor &O, const DEGNNDescentNeighbor &A, const DEGNNDescentNeighbor &B)
            {
                return (A.geo_distance_ - O.geo_distance_) * (B.emb_distance_ - O.emb_distance_) - (A.emb_distance_ - O.emb_distance_) * (B.geo_distance_ - O.geo_distance_);
            }

            void findConvexHull(std::vector<DEGNNDescentNeighbor> &points, std::vector<DEGNNDescentNeighbor> &convex_hull, std::vector<DEGNNDescentNeighbor> &remain_points)
            {
                // Build the lower hull
                for (const auto &point : points)
                {
                    while (convex_hull.size() >= 2 && cross(convex_hull[convex_hull.size() - 2], convex_hull.back(), point) <= 0)
                    {
                        remain_points.push_back(convex_hull.back());
                        convex_hull.pop_back();
                    }
                    convex_hull.push_back(point);
                }
                // Remove the last point of the upper hull because it's the same as the first point of the lower hull
            }

            void init_queue(std::vector<DEGNNDescentNeighbor> &insert_points)
            {
                std::vector<DEGNNDescentNeighbor> skyline_result;
                std::vector<DEGNNDescentNeighbor> remain_points;
                int l = 0;
                while (!insert_points.empty())
                {
                    findSkyline(insert_points, skyline_result, remain_points);
                    // findConvexHull(insert_points, skyline_result, remain_points);
                    insert_points.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, true, l);
                    }
                    std::vector<DEGNNDescentNeighbor>().swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(remain_points);
                    l++;
                }
                num_layer = l;
            }

            void findSkyline(std::vector<DEGNNDescentNeighbor> &points, std::vector<DEGNNDescentNeighbor> &skyline, std::vector<DEGNNDescentNeighbor> &remain_points)
            {
                // Sort points by x-coordinate
                // Sweep to find skyline
                float min_emb_dis = std::numeric_limits<float>::max();
                for (const auto &point : points)
                {
                    if (point.emb_distance_ < min_emb_dis)
                    {
                        skyline.push_back(point);
                        min_emb_dis = point.emb_distance_;
                    }
                    else
                    {
                        remain_points.emplace_back(point);
                    }
                }
                // O(n)
            }

            void updateNeighbor(int &nk)
            {
                std::vector<DEGNNDescentNeighbor> skyline_result;
                std::vector<DEGNNDescentNeighbor> remain_points;
                std::vector<DEGNNDescentNeighbor> candidate;
                candidate.swap(pool);
                int l = 0;
                int k = 0;
                sort(candidate.begin(), candidate.end());
                bool updated = true;
                while (pool.size() < M && candidate.size() > 0)
                {
                    findSkyline(candidate, skyline_result, remain_points);
                    // findConvexHull(candidate, skyline_result, remain_points); // too slow
                    candidate.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.flag, l);
                        if (updated)
                        {
                            if (point.flag == true)
                            {
                                nk = k;
                                updated = false;
                            }
                            else
                            {
                                nk++;
                            }
                        }
                        k++;
                    }
                    std::vector<DEGNNDescentNeighbor>().swap(skyline_result);
                    std::vector<DEGNNDescentNeighbor>().swap(remain_points);
                    l++;
                }
                num_layer = l;
            }
        };

        typedef std::vector<skyline_descent> SkylineDEG;
        SkylineDEG skylineDEG_;

        template <typename KeyType, typename DataType>
        class MaxHeap
        {
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key < i2.key; // 修改这里以适应最大堆，父节点的键应大于等于子节点的键
                }
            };

            MaxHeap()
            {
                std::make_heap(v_.begin(), v_.end()); // 确保v_初始化为一个堆
            }

            const KeyType top_key()
            {
                if (v_.empty())                                                                     // 优化：使用empty()检查是否为空
                    throw std::runtime_error("[Error] Called top_key() operation with empty heap"); // 无元素时抛出异常
                return v_[0].key;
            }

            Item top()
            {
                if (v_.empty())                                                                 // 使用empty()检查是否为空
                    throw std::runtime_error("[Error] Called top() operation with empty heap"); // 无元素时抛出异常
                return v_[0];
            }
            // 返回堆顶元素 最大元素

            void pop()
            {
                if (v_.empty())                                                                 // 添加检查避免在空堆上操作
                    throw std::runtime_error("[Error] Called pop() operation with empty heap"); // 无元素时抛出异常
                std::pop_heap(v_.begin(), v_.end());                                            // 将最大元素移到末尾
                v_.pop_back();                                                                  // 移除末尾元素（原最大元素）
            }
            // 移除堆顶元素

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));     // 添加新元素
                std::push_heap(v_.begin(), v_.end()); // 重新调整堆
            }
            // 向堆中添加新元素

            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class DEG_FurtherFirst
        {
        public:
            DEG_FurtherFirst(DEGNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetDistance() const { return dist_; }
            inline DEGNode *GetNode() const { return node_; }
            bool operator<(const DEG_FurtherFirst &n) const
            {
                return (dist_ < n.GetDistance());
                // 距离较小的节点会被视为“优先级较小” 因此FurtherFirst用于构建最大堆
            }

        private:
            DEGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float dist_;
        };

        class DEG_CloserFirst
        {
        public:
            DEG_CloserFirst(DEGNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetDistance() const { return dist_; }
            inline DEGNode *GetNode() const { return node_; }
            bool operator<(const DEG_CloserFirst &n) const
            {
                return (dist_ > n.GetDistance());
                // 距离较小的节点会被视为“优先级较高” 因此CloserFirst用于构建最小堆
            }

        private:
            DEGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float dist_;
        };

        class DEG_GeoFurtherFirst
        {
        public:
            DEG_GeoFurtherFirst(DEGNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline DEGNode *GetNode() const { return node_; }
            bool operator<(const DEG_GeoFurtherFirst &n) const
            {
                return (geo_distance_ < n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ < n.GetEmbDistance()));
                // 距离较小的节点会被视为“优先级较小” 因此 FurtherFirst 用于构建最大堆
            }

        private:
            DEGNode *node_;
            float emb_distance_;
            float geo_distance_;
        };

        class DEG_GeoCloserFirst
        {
        public:
            DEG_GeoCloserFirst(DEGNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline DEGNode *GetNode() const { return node_; }
            bool operator<(const DEG_GeoCloserFirst &n) const
            {
                return (geo_distance_ > n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ > n.GetEmbDistance()));
                // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
            }

        private:
            DEGNode *node_;
            float emb_distance_;
            float geo_distance_;
        };

        DEGNode *DEG_enterpoint_ = nullptr;
        std::vector<DEGNode *> DEG_nodes_;
        std::vector<DEGNode *> DEG_enterpoints;
        std::vector<DEGNNDescentNeighbor> DEG_enterpoints_skyeline;

        std::mutex enterpoint_mutex;
        std::vector<unsigned> enterpoint_set;
        unsigned rnn_size;
        float *emb_center, *loc_center;
    };

    class Index : public NNDescent, public NSW, public HNSW, public SSG, public NSG, public DEG, public baseline4
    {
    public:
        explicit Index(float max_emb_dist, float max_spatial_dist)
        {
            e_dist_ = new E_Distance(max_emb_dist);
            s_dist_ = new E_Distance(max_spatial_dist);
            // max_emb_dist_ = max_emb_dist;
            // max_spatial_dist_ = max_spatial_dist;
        }

        ~Index()
        {
            delete e_dist_;
            delete s_dist_;
        }

        struct SimpleNeighbor
        {
            unsigned id;
            float distance;

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

            inline bool operator<(const SimpleNeighbor &other) const
            {
                return distance < other.distance;
                // 该优先队列实际上表现为一个最大堆
            }
        };

        // sorted
        typedef std::vector<std::vector<SimpleNeighbor>> FinalGraph;
        typedef std::vector<std::vector<unsigned>> LoadGraph;

        class VisitedList
        {
        public:
            VisitedList(unsigned size) : size_(size), mark_(1)
            {
                visited_ = new unsigned int[size_];
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }

            ~VisitedList() { delete[] visited_; }

            inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }

            inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }

            inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }

            inline void Reset()
            {
                if (++mark_ == 0)
                {
                    mark_ = 1;
                    memset(visited_, 0, sizeof(unsigned int) * size_);
                }
            }

            inline unsigned int *GetVisited() { return visited_; }

            inline unsigned int GetVisitMark() { return mark_; }

        private:
            unsigned int *visited_;
            unsigned int size_;
            unsigned int mark_;
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
        {
            // find the location to insert
            // 数组 addr 的大小由参数 K 指定
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance)
            {
                memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance)
            {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1)
            {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)
                    right = mid;
                else
                    left = mid;
            }
            // check equal ID

            while (left > 0)
            {
                if (addr[left].distance < nn.distance)
                    break;
                if (addr[left].id == nn.id)
                    return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)
                return K + 1;
            memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        float *getBaseEmbData() const
        {
            return base_emb_data_;
        }

        void setBaseEmbData(float *baseEmbData)
        {
            base_emb_data_ = baseEmbData;
        }

        float *getBaseLocData() const
        {
            return base_loc_data_;
        }

        void setBaseLocData(float *baseLocData)
        {
            base_loc_data_ = baseLocData;
        }

        float *getQueryEmbData() const
        {
            return query_emb_data_;
        }

        void setQueryEmbData(float *queryEmbData)
        {
            query_emb_data_ = queryEmbData;
        }

        float *getQueryLocData() const
        {
            return query_loc_data_;
        }

        void setQueryLocData(float *queryLocData)
        {
            query_loc_data_ = queryLocData;
        }

        float *getQueryWeightData() const
        {
            return query_alpha_;
        }

        void setQueryWeightData(float *queryWeightData)
        {
            query_alpha_ = queryWeightData;
        }

        unsigned int *getGroundData() const
        {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData)
        {
            ground_data_ = groundData;
        }

        unsigned int getBaseLen() const
        {
            return base_len_;
        }

        void setBaseLen(unsigned int baseLen)
        {
            base_len_ = baseLen;
        }

        unsigned int getQueryLen() const
        {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen)
        {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const
        {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen)
        {
            ground_len_ = groundLen;
        }

        unsigned int getBaseEmbDim() const
        {
            return base_emb_dim_;
        }

        unsigned int getBaseLocDim() const
        {
            return base_loc_dim_;
        }

        void setBaseEmbDim(unsigned int baseEmbDim)
        {
            base_emb_dim_ = baseEmbDim;
        }

        void setBaseLocDim(unsigned int baseLocDim)
        {
            base_loc_dim_ = baseLocDim;
        }

        unsigned int getQueryEmbDim() const
        {
            return query_emb_dim_;
        }

        unsigned int getQueryLocDim() const
        {
            return query_loc_dim_;
        }

        void setQueryEmbDim(unsigned int queryEmbDim)
        {
            query_emb_dim_ = queryEmbDim;
        }

        void setQueryLocDim(unsigned int queryLocDim)
        {
            query_loc_dim_ = queryLocDim;
        }

        unsigned int getGroundDim() const
        {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim)
        {
            ground_dim_ = groundDim;
        }

        Parameters &getParam()
        {
            return param_;
        }

        void setParam(const Parameters &param)
        {
            param_ = param;
        }

        unsigned int getInitEdgesNum() const
        {
            return init_edges_num;
        }

        void setInitEdgesNum(unsigned int initEdgesNum)
        {
            init_edges_num = initEdgesNum;
        }

        unsigned int getCandidatesEdgesNum() const
        {
            return candidates_edges_num;
        }

        unsigned int getUpdateLayerNum() const
        {
            return update_layer_num;
        }

        void setCandidatesEdgesNum(unsigned int candidatesEdgesNum)
        {
            candidates_edges_num = candidatesEdgesNum;
        }

        void setUpdateLayerNum(unsigned int updatelayernum)
        {
            update_layer_num = updatelayernum;
        }

        unsigned int getResultEdgesNum() const
        {
            return result_edges_num;
        }

        void setResultEdgesNum(unsigned int resultEdgesNum)
        {
            result_edges_num = resultEdgesNum;
        }

        void set_alpha(float alpha)
        {
            // alpha_ = alpha;
            param_.set<unsigned>("alpha", alpha);
        }

        float get_alpha() const
        {
            return param_.get<float>("alpha");
        }
        E_Distance *get_E_Dist() const
        {
            return e_dist_;
        }

        // void Init_R_Tree(float max_s_dist)
        // {
        //     rtree_index = new RTreeIndex(max_s_dist);
        // }

        RTreeIndex &get_R_Tree()
        {
            return rtree_index;
        }

        E_Distance *get_S_Dist() const
        {
            return s_dist_;
        }

        FinalGraph &getFinalGraph()
        {
            return final_graph_;
        }

        SkylineDEG &getSkylineGraph()
        {
            return skylineDEG_;
        }

        LoadGraph &getLoadGraph()
        {
            return load_graph_;
        }

        LoadGraph &getExactGraph()
        {
            return exact_graph_;
        }

        TYPE getCandidateType() const
        {
            return candidate_type;
        }

        void setCandidateType(TYPE candidateType)
        {
            candidate_type = candidateType;
        }

        TYPE getPruneType() const
        {
            return prune_type;
        }

        void setPruneType(TYPE pruneType)
        {
            prune_type = pruneType;
        }

        TYPE getEntryType() const
        {
            return entry_type;
        }

        void setEntryType(TYPE entryType)
        {
            entry_type = entryType;
        }

        void setConnType(TYPE connType)
        {
            conn_type = connType;
        }

        TYPE getConnType() const
        {
            return conn_type;
        }

        unsigned int getDistCount() const
        {
            return dist_count;
        }

        void resetDistCount()
        {
            dist_count = 0;
        }

        void addDistCount()
        {
            dist_count += 1;
        }

        unsigned int getHopCount() const
        {
            return hop_count;
        }

        void resetHopCount()
        {
            hop_count = 0;
        }

        void addHopCount()
        {
            hop_count += 1;
        }

        void setNumThreads(const unsigned numthreads)
        {
            omp_set_num_threads(numthreads);
        }

        int i = 0;
        bool debug = false;

    private:
        float *base_emb_data_, *base_loc_data_, *query_emb_data_, *query_loc_data_, *query_alpha_;
        unsigned *ground_data_;

        unsigned base_len_, query_len_, ground_len_;
        unsigned base_emb_dim_, base_loc_dim_, query_emb_dim_, query_loc_dim_, ground_dim_;

        Parameters param_;
        unsigned init_edges_num;       // S
        unsigned candidates_edges_num; // L
        unsigned result_edges_num;     // K
        unsigned update_layer_num;

        E_Distance *e_dist_;
        E_Distance *s_dist_;

        FinalGraph final_graph_;
        LoadGraph load_graph_;
        LoadGraph exact_graph_;

        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        RTreeIndex rtree_index;

        unsigned dist_count = 0;
        unsigned hop_count = 0;

        float alpha_;
        float max_emb_dist_, max_spatial_dist_;
    };
}

#endif