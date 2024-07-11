//
// Created by MurphySL on 2020/10/23.
//

#include "component.h"

namespace stkq
{
    void ComponentSearchRouteGreedy::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                std::vector<unsigned int> &res)
    {
        const auto L = index->getParam().get<unsigned>("L_search");
        // 搜索过程中考虑的候选点数量 ef_search
        const auto K = index->getParam().get<unsigned>("K_search");
        // 最终需要返回的近邻数量
        std::vector<char> flags(index->getBaseLen(), 0);
        // 创建一个标志数组flags，用于标记已经访问过的点，避免重复处理。数组大小与数据集的基础长度（index->getBaseLen()）相同
        int k = 0;
        // 使用变量k从头到L遍历候选池pool中的点
        while (k < (int)L)
        {
            int nk = L;

            if (pool[k].flag)
            {
                pool[k].flag = false;
                unsigned n = pool[k].id;
                // 遍历当前点n的所有邻居。
                // 使用flags数组检查每个邻居是否已经被访问过，若未访问则标记为已访问
                index->addHopCount();
                for (unsigned m = 0; m < index->getLoadGraph()[n].size(); ++m)
                {
                    unsigned id = index->getLoadGraph()[n][m];

                    if (flags[id])
                        continue;
                    flags[id] = 1;

                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
                                                             index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                             index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
                                                             index->getBaseLocData() + id * index->getBaseLocDim(),
                                                             index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    index->addDistCount();

                    if (dist >= pool[L - 1].distance)
                        continue;
                    // 如果计算出的距离dist小于候选池中最远点的距离pool[L - 1].distance，则尝试将当前邻居插入候选池
                    Index::Neighbor nn(id, dist, true);
                    int r = Index::InsertIntoPool(pool.data(), L, nn);
                    // 使用Index::InsertIntoPool函数进行插入操作
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk)
                        nk = r;
                    // 更新最小插入位置nk
                }
                // lock to here
            }
            if (nk <= k)
            {
                k = nk;
                // 这个条件检查是否有元素被插入到候选池中的一个比当前索引 k 小或等的位置
                // 如果是这样，意味着在当前索引之前的某个位置有更新（新的邻居被插入）
                // 因此需要重新评估从这个新的位置（nk）开始的候选池，因为新插入的邻居可能改变了后续的搜索动态
                // 如果上述条件为真，说明我们需要将搜索的焦点移动回较早的位置 nk，因为候选池在这个位置有更新，可能出现了更好的候选邻居
                // 实际上, 因为已经访问过的节点flag为false, 所以实际上是先访问新插入节点的neighbor
            }
            else
            {
                ++k;
                // 根据插入操作的结果更新循环变量k
                // 如果nk小于或等于k，则更新k为nk，否则递增k
                // 如果 nk 大于 k，这意味着没有在当前位置或之前的位置插入新的邻居，我们可以安全地继续向前移动到候选池的下一个位置
            }
        }

        res.resize(K);
        for (size_t i = 0; i < K; i++)
        {
            res[i] = pool[i].id;
        }
    }

    void ComponentSearchRouteNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &res)
    {
        const auto K = index->getParam().get<unsigned>("K_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen());

        Index::HnswNode *enterpoint = index->nodes_[0];
        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        SearchAtLayer(query, enterpoint, 0, visited_list, result);

        while (!result.empty())
        {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }
        delete visited_list;
    }

    void ComponentSearchRouteNSW::SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                                                Index::VisitedList *visited_list,
                                                std::priority_queue<Index::FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;

        float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                 index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());

        float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

        index->addDistCount();
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty())
        {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);

                    e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                       index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbDim());

                    s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                       index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocDim());

                    d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    index->addDistCount();
                    if (result.size() < L || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > L)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentSearchRouteHNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res)
    {

        const auto K = index->getParam().get<unsigned>("K_search"); // 获取K_search参数来确定搜索结果的数量

        // const auto L = index->getParam().get<unsigned>("L_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen()); // 初始化一个VisitedList对象来跟踪已访问的节点

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::vector<std::pair<Index::HnswNode *, float>> ensure_k_path_; // 记录在每一层找到的最近节点及其距离

        Index::HnswNode *cur_node = enterpoint;
        float alpha = index->get_alpha();
        float e_d, s_d;
        if (alpha != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (alpha != 1)
        {
            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
                                               index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

        float d = alpha * e_d + (1 - alpha) * s_d;

        index->addDistCount();
        float cur_dist = d;

        ensure_k_path_.clear();
        ensure_k_path_.emplace_back(cur_node, cur_dist);

        for (auto i = index->max_level_; i >= 0; --i)
        {
            visited_list->Reset();
            unsigned visited_mark = visited_list->GetVisitMark();
            unsigned int *visited = visited_list->GetVisited();
            visited[cur_node->GetId()] = visited_mark;

            bool changed = true;
            while (changed)
            {
                changed = false;
                std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                index->addHopCount();
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                {
                    if (visited[(*iter)->GetId()] != visited_mark)
                    {
                        visited[(*iter)->GetId()] = visited_mark;

                        if (alpha != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }
                        if (alpha != 1)
                        {

                            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }
                        d = alpha * e_d + (1 - alpha) * s_d;

                        index->addDistCount();
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                            ensure_k_path_.emplace_back(cur_node, cur_dist);
                        }
                    }
                }
            }
        }

        // std::cout << "ensure_k : " << ensure_k_path_.size() << " " << ensure_k_path_[0].first->GetId() << std::endl;

        // std::vector<std::pair<Index::HnswNode*, float>> tmp;
        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        while (result.size() < K && !ensure_k_path_.empty())
        {
            cur_dist = ensure_k_path_.back().second;
            SearchAtLayer(query, ensure_k_path_.back().first, 0, visited_list, result);
            ensure_k_path_.pop_back();
        }

        while (!result.empty())
        {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }

        delete visited_list;
    }

    void ComponentSearchRouteHNSW::SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                                                 Index::VisitedList *visited_list,
                                                 std::priority_queue<Index::FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");

        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;

        float alpha = index->get_alpha();
        float e_d, s_d;
        if (alpha != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {

            e_d = 0;
        }
        if (alpha != 1)
        {

            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                               index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }
        float d = alpha * e_d + (1 - alpha) * s_d;

        index->addDistCount();
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty())
        {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    if (alpha != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (alpha != 1)
                    {
                        s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                           index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
                    d = alpha * e_d + (1 - alpha) * s_d;

                    index->addDistCount();
                    if (result.size() < L || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > L)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentSearchRouteGeoGraph::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                  std::vector<unsigned int> &res)
    {
        const auto K = index->getParam().get<unsigned>("K_search");
        auto *visited_list = new Index::VisitedList(index->getBaseLen());
        float alpha = index->get_alpha();
        visited_list->Reset();
        unsigned visited_mark = visited_list->GetVisitMark();
        unsigned int *visited = visited_list->GetVisited();
        // Index::GeoGraphNode *cur_node = index->geograph_enterpoint_;
        // Index::GeoGraphNode *cur_node = index->geograph_nodes_[10000];

        // float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
        //                                          index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
        //                                          index->getBaseEmbDim());

        // float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
        //                                          index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
        //                                          index->getBaseLocDim());

        // float cur_dist = alpha * e_d + (1 - alpha) * s_d;
        // std::vector<std::pair<Index::GeoGraphNode *, float>> ensure_k_path_;
        // ensure_k_path_.clear();
        // ensure_k_path_.emplace_back(cur_node, cur_dist);
        // index->addDistCount();
        // visited[cur_node->GetId()] = visited_mark;

        // e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + index->geograph_enterpoint_->GetId() * index->getBaseEmbDim(),
        //                                                 index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
        //                                                 index->getBaseEmbDim());

        // s_d = index->get_S_Dist()->compare(index->getBaseLocData() + index->geograph_enterpoint_->GetId() * index->getBaseLocDim(),
        //                                                 index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
        //                                                 index->getBaseLocDim());

        // std::cout << "e_d: " << e_d << "s_d: " << s_d << std::endl;

        // exit(1);

        // const std::vector<unsigned> &enterpoint_set = index->enterpoint_set;
        // for (int i = 0; i < enterpoint_set.size(); i++)
        // {
        //     cur_node = index->geograph_nodes_[enterpoint_set[i]];

        //     visited[cur_node->GetId()] = visited_mark;

        //     float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
        //                                              index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
        //                                              index->getBaseEmbDim());

        //     float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
        //                                              index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
        //                                              index->getBaseLocDim());

        //     float dist = alpha * e_d + (1 - alpha) * s_d;
        //     if (dist < cur_dist){
        //         ensure_k_path_.emplace_back(cur_node, dist);
        //         cur_dist = dist;
        //     }
        //     index->addDistCount();
        // }

        // cur_node = ensure_k_path_.back().first;

        // bool changed = true;
        // while (changed)
        // {
        //     changed = false;
        //     std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
        //     const std::vector<Index::GeoGraphNeighbor> &neighbors = cur_node->GetFriends(0);
        //     index->addHopCount();
        //     for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
        //     {
        //         if (iter->layer_ != 0)
        //         {
        //             break;
        //         }

        //         const std::vector<std::pair<float, float>> &use_range = iter->available_range;
        //         bool search_flag = false;
        //         for (int i = 0; i < use_range.size(); i++)
        //         {
        //             if (alpha >= use_range[i].first && alpha <= use_range[i].second)
        //             {
        //                 search_flag = true;
        //                 break;
        //             }
        //             if (alpha < use_range[i].first)
        //             {
        //                 break;
        //             }
        //             if (alpha > use_range[i].second)
        //             {
        //                 continue;
        //             }
        //         }
        //         if (search_flag)
        //         {
        //             if (visited[iter->id_] != visited_mark)
        //             {
        //                 visited[iter->id_] = visited_mark;

        //                 float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
        //                                                          index->getBaseEmbData() + iter->id_ * index->getBaseEmbDim(),
        //                                                          index->getBaseEmbDim());

        //                 float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
        //                                                          index->getBaseLocData() + iter->id_ * index->getBaseLocDim(),
        //                                                          index->getBaseLocDim());

        //                 float d = alpha * e_d + (1 - alpha) * s_d;

        //                 index->addDistCount();
        //                 if (d < cur_dist)
        //                 {
        //                     cur_dist = d;
        //                     cur_node = index->geograph_nodes_[iter->id_];
        //                     changed = true;
        //                     ensure_k_path_.emplace_back(cur_node, cur_dist);
        //                 }
        //             }
        //         }
        //     }
        // }

        // for (auto i = index->max_level_; i >= 0; --i)
        // {
        //     visited_list->Reset();
        //     unsigned visited_mark = visited_list->GetVisitMark();
        //     unsigned int *visited = visited_list->GetVisited();
        //     visited[cur_node->GetId()] = visited_mark;

        //     bool changed = true;
        //     while (changed)
        //     {
        //         changed = false;
        //         std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
        //         const std::vector<Index::GeoGraphNeighbor> &neighbors = cur_node->GetFriends(i);
        //         index->addHopCount();
        //         for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
        //         {
        //             const std::vector<std::pair<float, float>> &use_range = iter->available_range;
        //             bool search_flag = isInRange(index->get_alpha(), use_range);
        //             if (search_flag)
        //             {
        //                 if (visited[iter->id_] != visited_mark)
        //                 {
        //                     visited[iter->id_] = visited_mark;

        //                     e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + query * index->getBaseEmbDim(),
        //                                                        index->getBaseEmbData() + iter->id_ * index->getBaseEmbDim(),
        //                                                        index->getBaseEmbDim());

        //                     s_d = index->get_S_Dist()->compare(index->getQueryLocData() + query * index->getBaseLocDim(),
        //                                                        index->getBaseLocData() + iter->id_ * index->getBaseLocDim(),
        //                                                        index->getBaseLocDim());

        //                     d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

        //                     index->addDistCount();
        //                     if (d < cur_dist)
        //                     {
        //                         cur_dist = d;
        //                         cur_node = index->geograph_nodes_[iter->id_];
        //                         changed = true;
        //                         ensure_k_path_.emplace_back(cur_node, cur_dist);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        std::priority_queue<Index::GeoGraph_FurtherFirst> result;
        std::priority_queue<Index::GeoGraph_CloserFirst> tmp;

        // while (result.size() < K && !ensure_k_path_.empty())
        // {
        // cur_dist = ensure_k_path_.back().second;
        SearchAtLayer(query, index->geograph_enterpoint_, 0, visited_list, result);
        // ensure_k_path_.pop_back();
        // }

        while (!result.empty())
        {
            tmp.push(Index::GeoGraph_CloserFirst(result.top().GetNode(), result.top().GetEmbDistance(), result.top().GetLocDistance(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }

        delete visited_list;
    }

    // void ComponentSearchRouteGeoGraph::UpdateEnterpointSet()
    // {
    //     int K = 10; // enterpoint number
    //     std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex);
    //     std::vector<Index::GeoGraphNode *> enterpoints;

    //     enterpoints.push_back(index->geograph_nodes_[index->enterpoint_set[0]]);

    //     std::vector<int> hop_count(index->geograph_nodes_.size(), std::numeric_limits<int>::max() - 1);
    //     hop_count[index->enterpoint_set[0]] = 0;

    //     while (enterpoints.size() < K)
    //     {
    //         std::queue<Index::GeoGraphNode *> to_visit;
    //         to_visit.push(enterpoints.back());

    //         while (!to_visit.empty())
    //         {
    //             Index::GeoGraphNode *current_node = to_visit.front();
    //             to_visit.pop();
    //             int current_hop = hop_count[current_node->GetId()];

    //             for (auto &neighbor : current_node->GetFriends())
    //             {
    //                 Index::GeoGraphNode *neighbor_node = index->geograph_nodes_[neighbor.id_];
    //                 int new_hop_count = current_hop + 1;

    //                 if (new_hop_count < hop_count[neighbor_node->GetId()])
    //                 {
    //                     hop_count[neighbor_node->GetId()] = new_hop_count;
    //                     to_visit.push(neighbor_node);
    //                 }
    //             }
    //         }

    //         // 找到距离最大化的节点作为新的中心点
    //         int max_distance = 0;
    //         Index::GeoGraphNode *new_center = nullptr;
    //         for (size_t i = 0; i < index->enterpoint_set.size(); ++i)
    //         {
    //             int enterpoint_id = index->enterpoint_set[i];

    //             if (hop_count[enterpoint_id] > max_distance && hop_count[enterpoint_id] != std::numeric_limits<int>::max())
    //             {
    //                 max_distance = hop_count[enterpoint_id];
    //                 new_center = index->geograph_nodes_[enterpoint_id];
    //             }

    //             if (hop_count[i] == std::numeric_limits<int>::max())
    //             {
    //                 break;
    //             }
    //         }

    //         if (new_center != nullptr)
    //         {
    //             enterpoints.push_back(new_center);
    //             // 更新新enterpoint的hop_count为0
    //             hop_count[new_center->GetId()] = 0;
    //         }
    //         else
    //         {
    //             break; // 如果找不到新的中心点，退出循环
    //         }
    //     }
    //     index->geograph_enterpoints = enterpoints;
    // }

    void ComponentSearchRouteGeoGraph::SearchAtLayer(unsigned qnode, Index::GeoGraphNode *enterpoint, int level,
                                                     Index::VisitedList *visited_list,
                                                     std::priority_queue<Index::GeoGraph_FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");

        std::priority_queue<Index::GeoGraph_CloserFirst> candidates;
        float alpha = index->get_alpha();
        visited_list->Reset();

        // std::vector<unsigned> nearby_enterpoint;

        // Index::GeoGraphNode *cur_node = enterpoint;
        // float cur_e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
        //                                              index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
        //                                              index->getBaseEmbDim());

        // float cur_s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
        //                                              index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
        //                                              index->getBaseLocDim());

        // float cur_dist = alpha * cur_e_d + (1 - alpha) * cur_s_d;

        // bool changed = true;
        // while (changed)
        // {
        //     changed = false;
        //     std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
        //     const std::vector<Index::GeoGraphNeighbor> &neighbors = cur_node->GetFriends(0);
        //     index->addHopCount();
        //     for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
        //     {
        //         if (iter->layer_ != 0)
        //         {
        //             break;
        //         }

        //         const std::vector<std::pair<float, float>> &use_range = iter->available_range;
        //         bool search_flag = false;
        //         for (int i = 0; i < use_range.size(); i++)
        //         {
        //             if (alpha >= use_range[i].first && alpha <= use_range[i].second)
        //             {
        //                 search_flag = true;
        //                 break;
        //             }
        //             if (alpha < use_range[i].first)
        //             {
        //                 break;
        //             }
        //             if (alpha > use_range[i].second)
        //             {
        //                 continue;
        //             }
        //         }
        //         if (search_flag)
        //         {
        //             if (visited_list->NotVisited(iter->id_))
        //             {

        //                 visited_list->MarkAsVisited(iter->id_);

        //                 float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
        //                                                          index->getBaseEmbData() + iter->id_ * index->getBaseEmbDim(),
        //                                                          index->getBaseEmbDim());

        //                 float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
        //                                                          index->getBaseLocData() + iter->id_ * index->getBaseLocDim(),
        //                                                          index->getBaseLocDim());

        //                 float d = alpha * e_d + (1 - alpha) * s_d;

        //                 index->addDistCount();
        //                 if (d < cur_dist)
        //                 {
        //                     cur_dist = d;
        //                     cur_e_d = e_d;
        //                     cur_s_d = s_d;
        //                     cur_node = index->geograph_nodes_[iter->id_];
        //                     changed = true;
        //                 }
        //             }
        //         }
        //     }
        // }

        // index->addDistCount();
        // result.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
        // candidates.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
        // nearby_enterpoint.emplace_back(cur_node->GetId());

        bool m_first = false;
        // if (index->getBaseEmbDim() == index->getBaseLocDim())
        // {
        //     m_first = false;
        // }
        // else
        // {
        //     m_first = true;
        // }

        for (int i = 0; i < index->enterpoint_set.size(); i++)
        {
            Index::GeoGraphNode *cur_node = index->geograph_nodes_[index->enterpoint_set[i]];

            float cur_e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

            index->addDistCount();

            float cur_s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                         index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                                         index->getBaseLocDim());
            index->addDistCount();

            float cur_dist = alpha * cur_e_d + (1 - alpha) * cur_s_d;

            // changed = true;
            // while (changed)
            // {
            //     changed = false;
            //     std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
            //     const std::vector<Index::GeoGraphNeighbor> &neighbors = cur_node->GetFriends(0);
            //     index->addHopCount();
            //     for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
            //     {
            //         if (iter->layer_ != 0)
            //         {
            //             break;
            //         }

            //         const std::vector<std::pair<float, float>> &use_range = iter->available_range;
            //         bool search_flag = false;
            //         for (int i = 0; i < use_range.size(); i++)
            //         {
            //             if (alpha >= use_range[i].first && alpha <= use_range[i].second)
            //             {
            //                 search_flag = true;
            //                 break;
            //             }
            //             if (alpha < use_range[i].first)
            //             {
            //                 break;
            //             }
            //             if (alpha > use_range[i].second)
            //             {
            //                 continue;
            //             }
            //         }
            //         if (search_flag)
            //         {
            //             if (visited_list->NotVisited(iter->id_))
            //             {
            //                 visited_list->MarkAsVisited(iter->id_);

            //                 float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
            //                                                          index->getBaseEmbData() + iter->id_ * index->getBaseEmbDim(),
            //                                                          index->getBaseEmbDim());

            //                 float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
            //                                                          index->getBaseLocData() + iter->id_ * index->getBaseLocDim(),
            //                                                          index->getBaseLocDim());

            //                 float d = alpha * e_d + (1 - alpha) * s_d;

            //                 index->addDistCount();
            //                 if (d < cur_dist)
            //                 {
            //                     cur_dist = d;
            //                     cur_e_d = e_d;
            //                     cur_s_d = s_d;
            //                     cur_node = index->geograph_nodes_[iter->id_];
            //                     changed = true;
            //                 }
            //             }
            //         }
            //     }
            // }

            // index->addDistCount();
            // if (result.size() < 1 || result.top().GetDistance() > cur_dist)
            // {
            result.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
            candidates.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
            // nearby_enterpoint.emplace_back(cur_node->GetId());
            // }
            // result.emplace(index->geograph_nodes_[index->enterpoint_set[i]], e_d, s_d, d);
            // candidates.emplace(index->geograph_nodes_[index->enterpoint_set[i]], e_d, s_d, d);
            visited_list->MarkAsVisited(cur_node->GetId());
        }

        // visited_list->Reset();
        // for (auto &id: nearby_enterpoint){
        //     visited_list->MarkAsVisited(id);
        // }
        // step 1 search on layer 0
        // const Index::GeoGraph_CloserFirst &candidate_node = candidates.top();
        // while (!candidates.empty())
        // {
        //     candidates.pop();
        // }
        // candidates.push(candidate_node);

        while (!candidates.empty())
        {
            const Index::GeoGraph_CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::GeoGraphNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            // const std::vector<std::shared_ptr<Index::GeoGraphEdge>> &neighbors = candidate_node->GetFriends();
            std::vector<Index::GeoGraphSimpleNeighbor> &neighbors = candidate_node->GetSearchFriends();
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int neighbor_id = neighbor.id_;
                const std::vector<std::pair<float, float>> &use_range = neighbor.available_range;
                bool search_flag = false;
                for (int i = 0; i < use_range.size(); i++)
                {
                    if (alpha >= use_range[i].first && alpha <= use_range[i].second)
                    {
                        search_flag = true;
                        break;
                    }
                    if (alpha < use_range[i].first)
                    {
                        break;
                    }
                    if (alpha > use_range[i].second)
                    {
                        continue;
                    }
                }
                
                // search_flag = true;

                if (search_flag)
                {
                    if (visited_list->NotVisited(neighbor_id))
                    {
                        visited_list->MarkAsVisited(neighbor_id);

                        if (result.size() >= L)
                        {
                            if (m_first)
                            {
                                float threshold = result.top().GetDistance();

                                float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                                         index->getBaseLocData() + neighbor_id * index->getBaseLocDim(),
                                                                         index->getBaseLocDim());

                                if ((1 - alpha) * s_d >= threshold)
                                {
                                    continue;
                                }

                                float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                                         index->getBaseEmbData() + neighbor_id * index->getBaseEmbDim(),
                                                                         index->getBaseEmbDim());
                                index->addDistCount();

                                float d = alpha * e_d + (1 - alpha) * s_d;

                                if (threshold > d)
                                {
                                    result.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                    candidates.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                    if (result.size() > L)
                                        result.pop();
                                }
                            }
                            else
                            {
                                float threshold = result.top().GetDistance();

                                if (alpha <= 0.5)
                                {
                                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                                             index->getBaseLocData() + neighbor_id * index->getBaseLocDim(),
                                                                             index->getBaseLocDim());

                                    if ((1 - alpha) * s_d >= threshold)
                                    {
                                        continue;
                                    }

                                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                                             index->getBaseEmbData() + neighbor_id * index->getBaseEmbDim(),
                                                                             index->getBaseEmbDim());
                                    index->addDistCount();

                                    float d = alpha * e_d + (1 - alpha) * s_d;

                                    if (threshold > d)
                                    {
                                        result.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                        candidates.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                        if (result.size() > L)
                                            result.pop();
                                    }
                                }
                                else
                                {
                                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                                             index->getBaseEmbData() + neighbor_id * index->getBaseEmbDim(),
                                                                             index->getBaseEmbDim());

                                    if (alpha * e_d >= threshold)
                                    {
                                        continue;
                                    }

                                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                                             index->getBaseLocData() + neighbor_id * index->getBaseLocDim(),
                                                                             index->getBaseLocDim());

                                    index->addDistCount();

                                    float d = alpha * e_d + (1 - alpha) * s_d;

                                    if (threshold > d)
                                    {
                                        result.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                        candidates.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                                        if (result.size() > L)
                                            result.pop();
                                    }
                                }
                            }
                        }
                        else
                        {
                            float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + qnode * index->getBaseLocDim(),
                                                                     index->getBaseLocData() + neighbor_id * index->getBaseLocDim(),
                                                                     index->getBaseLocDim());

                            float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + qnode * index->getBaseEmbDim(),
                                                                     index->getBaseEmbData() + neighbor_id * index->getBaseEmbDim(),
                                                                     index->getBaseEmbDim());
                            float d = alpha * e_d + (1 - alpha) * s_d;
                            result.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                            candidates.emplace(index->geograph_nodes_[neighbor_id], e_d, s_d, d);
                            if (result.size() > L)
                                result.pop();
                        }
                    }
                }
            }
        }
    }

    // void ComponentSearchRouteGuided::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
    //                                             std::vector<unsigned int> &res) {
    //     const auto L = index->getParam().get<unsigned>("L_search");
    //     const auto K = index->getParam().get<unsigned>("K_search");

    //     std::vector<char> flags(index->getBaseLen(), 0);

    //     int k = 0;
    //     while (k < (int)L) {
    //         int nk = L;

    //         if (pool[k].flag) {
    //             pool[k].flag = false;
    //             unsigned n = pool[k].id;

    //             unsigned div_dim_ = index->Tn[n].div_dim;
    //             unsigned left_len = index->Tn[n].left.size();
    //             // std::cout << "left_len: " << left_len << std::endl;
    //             unsigned right_len = index->Tn[n].right.size();
    //             // std::cout << "right_len: " << right_len << std::endl;
    //             std::vector<unsigned> nn;
    //             unsigned MaxM;
    //             if ((index->getQueryEmbData() + index->getQueryDim() * query)[div_dim_] < (index->getBaseEmbData() + index->getBaseDim() * n)[div_dim_]) {
    //                 MaxM = left_len;
    //                 nn = index->Tn[n].left;
    //             }
    //             else {
    //                 MaxM = right_len;
    //                 nn = index->Tn[n].right;
    //             }

    //             index->addHopCount();
    //             for (unsigned m = 0; m < MaxM; ++m) {
    //                 unsigned id = nn[m];
    //                 if (flags[id]) continue;
    //                 flags[id] = 1;
    //                 float dist = index->getDist()->compare(index->getQueryEmbData() + query * index->getQueryDim(),
    //                                                        index->getBaseEmbData() + id * index->getBaseDim(),
    //                                                        (unsigned)index->getBaseDim());
    //                 index->addDistCount();
    //                 if (dist >= pool[L - 1].distance) continue;
    //                 Index::Neighbor nn(id, dist, true);
    //                 int r = Index::InsertIntoPool(pool.data(), L, nn);

    //                 // if(L+1 < retset.size()) ++L;
    //                 if (r < nk) nk = r;
    //             }
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }

    //     res.resize(K);
    //     for (size_t i = 0; i < K; i++) {
    //         res[i] = pool[i].id;
    //     }
    // }

}