#ifndef STKQ_COMPONENT_H
#define STKQ_COMPONENT_H

#include "index.h"

namespace stkq
{
    class Component
    {
    public:
        explicit Component(Index *index) : index(index) {}
        virtual ~Component() { delete index; }

    protected:
        Index *index = nullptr;
    };

    // load data
    class ComponentLoad : public Component
    {
    public:
        explicit ComponentLoad(Index *index) : Component(index) {}

        virtual void LoadInner(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, Parameters &parameters);

        // virtual void load_partition(char *partition_file);
    };

    // initial graph
    class ComponentInit : public Component
    {
    public:
        explicit ComponentInit(Index *index) : Component(index) {}

        virtual void InitInner() = 0;
    };

    class ComponentInitBS4 : public ComponentInit
    {
    public:
        explicit ComponentInitBS4(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void Build();

        void Merge();

        static int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::BS4Node *qnode, Index::VisitedList *visited_list, unsigned iter);

        void SearchAtLayer(Index::BS4Node *qnode, Index::BS4Node *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::BS4FurtherFirst> &result);

        void Link(Index::BS4Node *source, Index::BS4Node *target, int level);
    };

    class ComponentInitRTree : public ComponentInit
    {
    public:
        explicit ComponentInitRTree(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();
    };

    class ComponentInitRandom : public ComponentInit
    {
    public:
        explicit ComponentInitRandom(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size);
    };

    // refine graph
    class ComponentRefine : public Component
    {
    public:
        explicit ComponentRefine(Index *index) : Component(index) {}

        virtual void RefineInner() = 0;
    };

    class ComponentRefineNNDescent : public ComponentRefine
    {
    public:
        explicit ComponentRefineNNDescent(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void init();

        void NNDescent();

        void join();

        void update();

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned>> &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned>> &acc_eval_set);
    };

    class ComponentRefineNSG : public ComponentRefine
    {
    public:
        explicit ComponentRefineNSG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, Index::SimpleNeighbor *cut_graph_);

        void SetConfigs();
    };

    class ComponentInitNSW : public ComponentInit
    {
    public:
        explicit ComponentInitNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitHNSW : public ComponentInit
    {
    public:
        explicit ComponentInitHNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void Build(bool reverse);

        static int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitDEG : public ComponentInit
    {
    public:
        explicit ComponentInitDEG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomNodeLevel();

        int GetRandomSeedPerThread();

        void BuildByIncrementInsert();

        void init();

        void EntryInner();

        void join();

        void update();

        void Refine();

        void SkylineNNDescent();

        void PruneInner(unsigned n, unsigned range,
                        std::vector<Index::DEGNeighbor> &cut_graph_);

        void findSkyline(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &skyline,
                         std::vector<Index::DEGNeighbor> &remain_points);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                         std::vector<std::vector<Index::DEGNeighbor>> &cut_graph_);

        void InsertNode(Index::DEGNode *qnode, Index::VisitedList *visited_list);

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N);

        void SearchAtLayer(Index::DEGNode *qnode,
                           Index::VisitedList *visited_list,
                           std::vector<Index::DEGNNDescentNeighbor> &result);

        void UpdateEnterpointSet(Index::DEGNode *qnode);
        void UpdateEnterpointSet();

        void Link(Index::DEGNode *source, Index::DEGNode *target, int level, float e_dist, float s_dist);

        bool isInRange(float alpha, const std::vector<std::pair<float, float>> &use_range)
        {
            // 遍历所有范围
            for (const auto &range : use_range)
            {
                // 检查alpha是否在当前范围内
                if (alpha >= range.first && alpha <= range.second)
                {
                    return true; // alpha在范围内
                }
                if (alpha < range.first)
                {
                    return false;
                }
                if (alpha > range.second)
                {
                    continue;
                }
            }
            // 没有找到alpha在任何范围内
            return false;
        }

        void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                          const std::pair<float, float> &use_range,
                          std::vector<std::pair<float, float>> &shared_use_range)
        {
            // 遍历picked_available_range中的每个范围
            // std::sort(picked_available_range.begin(), picked_available_range.end());
            for (const auto &range : picked_available_range)
            {
                // 计算交集的下限和上限
                if (range.second <= use_range.first)
                {
                    continue;
                }
                else if (range.first >= use_range.second)
                {
                    break;
                }
                else
                {
                    float lower_bound = std::max(range.first, use_range.first);
                    float upper_bound = std::min(range.second, use_range.second);
                    // 检查交集是否有效（下限小于等于上限）
                    if (lower_bound < upper_bound)
                    {
                        // 如果交集有效，添加到shared_use_range中
                        shared_use_range.push_back({lower_bound, upper_bound});
                    }
                }
            }
        }

        void get_use_range(std::vector<std::pair<float, float>> &prune_range, std::vector<std::pair<float, float>> &after_pruned_use_range)
        {
            if (prune_range.empty())
            {
                after_pruned_use_range.push_back(std::make_pair(0.0f, 1.0f));
                return;
            }
            if (prune_range[0].first > 0)
            {
                float gap = prune_range[0].first - 0;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(0, prune_range[0].first));
                }
            }
            int iter;
            for (iter = 0; iter < prune_range.size() - 1; iter++)
            {
                float gap = prune_range[iter + 1].first - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, prune_range[iter + 1].first));
                }
            }
            if (prune_range[iter].second < 1)
            {
                float gap = 1 - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, 1));
                }
            }
            return;
        };

        std::vector<std::pair<float, float>> get_use_range(std::vector<std::pair<float, float>> &intervals)
        {
            std::vector<std::pair<float, float>> use_range;
            if (intervals.empty())
            {
                use_range.push_back(std::make_pair(0, 1));
                return use_range;
            }
            if (intervals[0].first > 0 && intervals[0].first >= 0.01)
            {
                use_range.push_back(std::make_pair(0, intervals[0].first));
            }
            for (size_t i = 0; i < intervals.size() - 1; i++)
            {
                float gap = intervals[i + 1].first - intervals[i].second;
                if (gap >= 0.01)
                {
                    use_range.push_back(std::make_pair(intervals[i].second, intervals[i + 1].first));
                }
            }
            if (intervals[intervals.size() - 1].second < 1 && (1 - intervals[intervals.size() - 1].second) >= 0.01)
            {
                use_range.push_back(std::make_pair(intervals[intervals.size() - 1].second, 1));
            }
            return use_range;
        };

        std::vector<std::pair<float, float>> mergeIntervals(std::vector<std::pair<float, float>> &intervals)
        {
            if (intervals.empty())
                return {};

            // 先对区间按照起始值进行排序
            std::sort(intervals.begin(), intervals.end());

            std::vector<std::pair<float, float>> merged;

            merged.push_back(intervals[0]);

            for (size_t i = 1; i < intervals.size() - 1; i++)
            {
                if (intervals[i].first <= merged.back().second)
                {
                    // 如果当前区间的起始值小于等于前一个区间的终止值，则合并这两个区间
                    merged.back().second = std::max(merged.back().second, intervals[i].second);
                }
                else
                {
                    // 否则，当前区间与前一个区间不相交，将其添加到结果中
                    merged.push_back(intervals[i]);
                }
            }
            return merged;
        }
    };

    class ComponentPrune : public Component
    {
    public:
        explicit ComponentPrune(Index *index) : Component(index) {}
    };

    class ComponentPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentPruneHeuristic(Index *index) : ComponentPrune(index) {}

        // void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
        //                 std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_);

        void Hnsw2Neighbor(unsigned query, unsigned range, std::priority_queue<Index::FurtherFirst> &result)
        {
            // 它的作用是对给定节点的邻居列表进行剪枝，以选择最优的邻居
            int n = result.size();
            std::vector<Index::SimpleNeighbor> pool(n);
            // 创建一个向量 pool 用于存储与查询节点距离最近的邻居
            std::unordered_map<int, Index::HnswNode *> tmp;

            for (int i = n - 1; i >= 0; i--)
            {
                Index::FurtherFirst f = result.top(); // 最大堆
                pool[i] = Index::SimpleNeighbor(f.GetNode()->GetId(), f.GetDistance());
                tmp[f.GetNode()->GetId()] = f.GetNode();
                result.pop();
            }

            boost::dynamic_bitset<> flags; // 创建一个动态位集合，通常用于标记状态或记录已访问的节点

            auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * range]; // 动态分配一个数组，用于存储剪枝后的图数据, 类似前向星链

            PruneInner(query, range, flags, pool, cut_graph_);

            for (unsigned j = 0; j < range; j++)
            {
                if (cut_graph_[range * query + j].distance == -1)
                    break;

                result.push(Index::FurtherFirst(tmp[cut_graph_[range * query + j].id], cut_graph_[range * query + j].distance));
            }

            delete[] cut_graph_;

            std::vector<Index::SimpleNeighbor>().swap(pool);
            std::unordered_map<int, Index::HnswNode *>().swap(tmp);
            // 这两行代码通过交换技巧来清空 pool 向量和 tmp 哈希表 一种常用的释放容器占用内存的方法
        }
void Hnsw2Neighbor(unsigned query, unsigned range, std::priority_queue<Index::BS4FurtherFirst> &result)
        {
            // 它的作用是对给定节点的邻居列表进行剪枝，以选择最优的邻居
            int n = result.size();
            std::vector<Index::SimpleNeighbor> pool(n);
            // 创建一个向量 pool 用于存储与查询节点距离最近的邻居
            std::unordered_map<int, Index::BS4Node *> tmp;

            for (int i = n - 1; i >= 0; i--)
            {
                Index::BS4FurtherFirst f = result.top(); // 最大堆
                pool[i] = Index::SimpleNeighbor(f.GetNode()->GetId(), f.GetDistance());
                tmp[f.GetNode()->GetId()] = f.GetNode();
                result.pop();
            }

            boost::dynamic_bitset<> flags; // 创建一个动态位集合，通常用于标记状态或记录已访问的节点

            auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * range]; // 动态分配一个数组，用于存储剪枝后的图数据, 类似前向星链

            PruneInner(query, range, flags, pool, cut_graph_);

            for (unsigned j = 0; j < range; j++)
            {
                if (cut_graph_[range * query + j].distance == -1)
                    break;

                result.push(Index::BS4FurtherFirst(tmp[cut_graph_[range * query + j].id], cut_graph_[range * query + j].distance));
            }

            delete[] cut_graph_;

            std::vector<Index::SimpleNeighbor>().swap(pool);
            std::unordered_map<int, Index::BS4Node *>().swap(tmp);
            // 这两行代码通过交换技巧来清空 pool 向量和 tmp 哈希表 一种常用的释放容器占用内存的方法
        }
                
    };

    // graph conn
    class ComponentConn : public Component
    {
    public:
        explicit ComponentConn(Index *index) : Component(index) {}

        virtual void ConnInner() = 0;
    };

    class ComponentConnNSGDFS : ComponentConn
    {
    public:
        explicit ComponentConnNSGDFS(Index *index) : ComponentConn(index) {}

        void ConnInner();

    private:
        void tree_grow();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root);

        void
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const int query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    // select candidate
    class ComponentCandidate : public Component
    {
    public:
        explicit ComponentCandidate(Index *index) : Component(index) {}

        // virtual void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
        //                             std::vector<Index::SimpleNeighbor> &pool) = 0;
    };

    class ComponentCandidateNSG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateNSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result);
    };

    class ComponentCandidateDEG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateDEG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, std::vector<unsigned> enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::DEGNNDescentNeighbor> &result);
    };

    class ComponentDEGPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentDEGPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                          const std::pair<float, float> &use_range,
                          std::vector<std::pair<float, float>> &shared_use_range)
        {
            // 遍历picked_available_range中的每个范围
            // std::sort(picked_available_range.begin(), picked_available_range.end());
            for (const auto &range : picked_available_range)
            {
                // 计算交集的下限和上限
                if (range.second <= use_range.first)
                {
                    continue;
                }
                else if (range.first >= use_range.second)
                {
                    break;
                }
                else
                {
                    float lower_bound = std::max(range.first, use_range.first);
                    float upper_bound = std::min(range.second, use_range.second);
                    // 检查交集是否有效（下限小于等于上限）
                    if (lower_bound < upper_bound)
                    {
                        // 如果交集有效，添加到shared_use_range中
                        shared_use_range.push_back({lower_bound, upper_bound});
                    }
                }
            }
        }

        void get_use_range(std::vector<std::pair<float, float>> &prune_range, std::vector<std::pair<float, float>> &after_pruned_use_range)
        {
            if (prune_range.empty())
            {
                after_pruned_use_range.push_back(std::make_pair(0.0f, 1.0f));
                return;
            }
            if (prune_range[0].first > 0)
            {
                float gap = prune_range[0].first - 0;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(0, prune_range[0].first));
                }
            }
            int iter;
            for (iter = 0; iter < prune_range.size() - 1; iter++)
            {
                float gap = prune_range[iter + 1].first - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, prune_range[iter + 1].first));
                }
            }
            if (prune_range[iter].second < 1)
            {
                float gap = 1 - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, 1));
                }
            }
            return;
        };

        std::vector<std::pair<float, float>> get_use_range(std::vector<std::pair<float, float>> &intervals)
        {
            std::vector<std::pair<float, float>> use_range;
            if (intervals.empty())
            {
                use_range.push_back(std::make_pair(0, 1));
                return use_range;
            }
            if (intervals[0].first > 0 && intervals[0].first >= 0.01)
            {
                use_range.push_back(std::make_pair(0, intervals[0].first));
            }
            for (size_t i = 0; i < intervals.size() - 1; i++)
            {
                float gap = intervals[i + 1].first - intervals[i].second;
                if (gap >= 0.01)
                {
                    use_range.push_back(std::make_pair(intervals[i].second, intervals[i + 1].first));
                }
            }
            if (intervals[intervals.size() - 1].second < 1 && (1 - intervals[intervals.size() - 1].second) >= 0.01)
            {
                use_range.push_back(std::make_pair(intervals[intervals.size() - 1].second, 1));
            }
            return use_range;
        };

        std::vector<std::pair<float, float>> mergeIntervals(std::vector<std::pair<float, float>> &intervals)
        {
            if (intervals.empty())
                return {};

            // 先对区间按照起始值进行排序
            std::sort(intervals.begin(), intervals.end());

            std::vector<std::pair<float, float>> merged;

            merged.push_back(intervals[0]);

            for (size_t i = 1; i < intervals.size() - 1; i++)
            {
                if (intervals[i].first <= merged.back().second)
                {
                    // 如果当前区间的起始值小于等于前一个区间的终止值，则合并这两个区间
                    merged.back().second = std::max(merged.back().second, intervals[i].second);
                }
                else
                {
                    // 否则，当前区间与前一个区间不相交，将其添加到结果中
                    merged.push_back(intervals[i]);
                }
            }
            return merged;
        }

        float crossProduct(const Index::DEGNeighbor &O, const Index::DEGNeighbor &A, const Index::DEGNeighbor &B)
        {
            float result = (A.geo_distance_ - O.geo_distance_) * (B.emb_distance_ - O.emb_distance_) - (A.emb_distance_ - O.emb_distance_) * (B.geo_distance_ - O.geo_distance_);
            return result;
        };

        void lowerConvexHull(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &L)
        {
            // 构造下凸包
            std::vector<Index::DEGNeighbor> hull;
            for (int i = 0; i < points.size(); i++)
            {
                while (hull.size() >= 2 && (crossProduct(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0))
                {
                    hull.pop_back();
                    // 这段代码是用来判断是否需要替换掉前一个element
                }
                hull.push_back(points[i]);
            }
            L.push_back(hull[0]);
            for (int i = 1; i < hull.size(); ++i)
            {
                float deltaY = hull[i].emb_distance_ - hull[i - 1].emb_distance_;
                float deltaX = hull[i].geo_distance_ - hull[i - 1].geo_distance_;
                if (deltaX * deltaY > 0)
                {
                    break;
                }
                else
                {
                    L.push_back(hull[i]);
                }
            }
            // 构造上凸包之前的准备
            // 由于最后一个点既在下凸包也在上凸包中，所以先删除以避免重复
        };

        void findSkyline(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &skyline)
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
            }
            // O(n)
        }

        void PruneInner(std::vector<Index::DEGNNDescentNeighbor> &pool, unsigned range,
                        // std::vector<Index::DEGNeighbor> &picked);
                        std::vector<Index::DEGNeighbor> &cut_graph_);

        void DEG2Neighbor(unsigned qnode, unsigned range, std::vector<Index::DEGNNDescentNeighbor> &pool, std::vector<Index::DEGNeighbor> &result)
        {
            PruneInner(pool, range, result);
        };
    };

    class ComponentSearchRoute : public Component
    {
    public:
        explicit ComponentSearchRoute(Index *index) : Component(index) {}

        virtual void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) = 0;
    };

    class ComponentSearchRouteBS4 : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteBS4(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        // void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
        //                  size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
        void SearchAtLayer(unsigned qnode, Index::BS4Node *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::BS4FurtherFirst> &result);
    };

    class ComponentSearchRouteGreedy : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteGreedy(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };

    class ComponentSearchRouteNSW : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteHNSW : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteHNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        // void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
        //                  size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteDEG : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteDEG(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

        bool isInRange(float alpha, const std::vector<std::pair<float, float>> &use_range)
        {
            // 遍历所有范围
            for (const auto &range : use_range)
            {
                // 检查alpha是否在当前范围内
                if (alpha >= range.first && alpha <= range.second)
                {
                    return true; // alpha在范围内
                }
                if (alpha < range.first)
                {
                    return false;
                }
                if (alpha > range.second)
                {
                    continue;
                }
            }
            // 没有找到alpha在任何范围内
            return false;
        }

    private:
        // void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
        //                  size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
        void SearchAtLayer(unsigned qnode, Index::DEGNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::DEG_FurtherFirst> &result);
    };

    // search entry
    class ComponentSearchEntry : public Component
    {
    public:
        explicit ComponentSearchEntry(Index *index) : Component(index) {}

        virtual void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) = 0;
    };

    // class ComponentSearchEntryCentroid : public ComponentSearchEntry {
    // public:
    //     explicit ComponentSearchEntryCentroid(Index *index) : ComponentSearchEntry(index) {}

    //     void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    // };

    class ComponentSearchEntryNone : public ComponentSearchEntry
    {
    public:
        explicit ComponentSearchEntryNone(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    // entry
    class ComponentRefineEntry : public Component
    {
    public:
        explicit ComponentRefineEntry(Index *index) : Component(index) {}

        virtual void EntryInner() = 0;
    };

    class ComponentRefineEntryCentroid : public ComponentRefineEntry
    {
    public:
        explicit ComponentRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retSet, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::Neighbor> &retset,
                      std::vector<Index::Neighbor> &fullset);
    };

    class ComponentDEGRefineEntryCentroid : public ComponentRefineEntry
    {
    public:
        explicit ComponentDEGRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retSet, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::DEGNNDescentNeighbor> &retset,
                      std::vector<Index::DEGNNDescentNeighbor> &fullset);
    };

    class ComponentSearchEntryCentroid : public ComponentSearchEntry
    {
    public:
        explicit ComponentSearchEntryCentroid(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };
}

#endif