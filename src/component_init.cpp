#include "component.h"
#include <functional>

namespace stkq
{
    void ComponentInitRandom::InitInner()
    {
        SetConfigs(); // 设置配置

        unsigned range = index->getInitEdgesNum(); // 获取初始化边的数量

        index->getFinalGraph().resize(index->getBaseLen()); // 根据基础长度调整最终图的大小

        std::mt19937 rng(rand()); // 使用随机数生成器

#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {                                             // 遍历基础长度
            index->getFinalGraph()[i].reserve(range); // 为每个节点预留边的空间
            std::vector<unsigned> tmp(range);         // 创建临时向量存储随机生成的边
            GenRandom(rng, tmp.data(), range);        // 生成随机边
            // 为每个基础节点生成随机连接的节点列表
            for (unsigned j = 0; j < range; j++)
            {
                unsigned id = tmp[j]; // 获取随机生成的节点ID
                if (id == i)
                {
                    continue; // 如果随机生成的ID与当前节点相同，则跳过
                }

                float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + i * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

                float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + i * index->getBaseLocDim(),
                                                         index->getBaseLocData() + id * index->getBaseLocDim(),
                                                         index->getBaseLocDim());

                float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                index->getFinalGraph()[i].emplace_back(id, dist);
            }
            std::sort(index->getFinalGraph()[i].begin(), index->getFinalGraph()[i].end());
        }
    }

    void ComponentInitRandom::SetConfigs()
    {
        index->setInitEdgesNum(index->getParam().get<unsigned>("S"));
        // 设置初始化边的数量 S is set to be 10
    }

    void ComponentInitRandom::GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size)
    {
        unsigned N = index->getBaseLen();

        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size); // 对于addr数组的每个元素，生成一个在0到N - size之间的随机数
        }
        std::sort(addr, addr + size); // 将addr数组中的随机数进行升序排序

        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        // 遍历数组，确保每个随机数都是唯一的。如果当前随机数不大于前一个随机数，则将其设置为前一个随机数加一。这一步骤保证了即使初始随机数有重复
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
        // 生成一个0到N-1之间的随机偏移量off，然后将这个偏移量应用到数组中的每个元素上
        // 通过取模N确保结果仍在合法范围内。这样做的目的是在保持数字唯一性的同时，增加随机性
    }

    // NSW
    void ComponentInitNSW::InitInner()
    {
        SetConfigs();
        index->nodes_.resize(index->getBaseLen());
        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    // NSWV2
    void ComponentInitNSWV2::InitInner()
    {
        SetConfigs();
        index->nodes_.resize(index->getBaseLen());
        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    void ComponentInitNSW::SetConfigs()
    {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
    }

    void ComponentInitNSWV2::SetConfigs()
    {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
    }

    void ComponentInitNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list)
    {
        Index::HnswNode *enterpoint = index->enterpoint_;

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        // CANDIDATE
        SearchAtLayer(qnode, enterpoint, 0, visited_list, result);

        while (!result.empty())
        {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        int pos = 0;
        while (!tmp.empty() && pos < index->NN_)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            Link(top_node, qnode, 0);
            Link(qnode, top_node, 0);
            pos++;
        }
    }

    void ComponentInitNSWV2::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list)
    {
        Index::HnswNode *enterpoint = index->enterpoint_;

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        std::vector<float> selected_alpha = {0.1, 0.3, 0.5, 0.7, 0.9};
        for (int i = 0; i < selected_alpha.size(); i++)
        {
            // CANDIDATE
            float alpha = selected_alpha[i];
            SearchAtLayer(qnode, enterpoint, 0, visited_list, result, alpha);

            while (!result.empty())
            {
                tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
                result.pop();
            }

            int pos = 0;
            while (!tmp.empty() && pos < index->NN_ / 5)
            {
                auto *top_node = tmp.top().GetNode();
                tmp.pop();
                Link(top_node, qnode, 0);
                Link(qnode, top_node, 0);
                pos++;
            }
        }
    }

    void ComponentInitNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result)
    {
        std::priority_queue<Index::CloserFirst> candidates;
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());
        float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

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
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbDim());

                    s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocDim());
                    d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitNSWV2::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                           Index::VisitedList *visited_list,
                                           std::priority_queue<Index::FurtherFirst> &result, float alpha)
    {
        std::priority_queue<Index::CloserFirst> candidates;
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());
        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());
        float d = alpha * e_d + (1 - alpha) * s_d;

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
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbDim());
                    s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocDim());
                    d = alpha * e_d + (1 - alpha) * s_d;

                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        source->AddFriends(target, true);
        // target->AddFriends(source, true);

        // std::unique_lock<std::mutex> lock(source->GetAccessGuard()); // 使用互斥锁确保在多线程环境下对源节点 source 的访问是线程安全的
        // std::vector<Index::HnswNode *> &neighbors = source->GetFriends(0);
        // neighbors.push_back(target);
        // //  获取源节点的邻居列表并添加目标节点
        // bool shrink = neighbors.size() > index->NN_;

        // if (!shrink)
        //     return;

        // std::priority_queue<Index::FurtherFirst> tempres;
        // for (const auto &neighbor : neighbors)
        // {
        //     float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + source->GetId() * index->getBaseEmbDim(),
        //                                              index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
        //                                              index->getBaseEmbDim());

        //     float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + source->GetId() * index->getBaseLocDim(),
        //                                              index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
        //                                              index->getBaseLocDim());

        //     float tmp = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

        //     tempres.push(Index::FurtherFirst(neighbor, tmp));
        // }

        // // PRUNE
        // ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        // a->Hnsw2Neighbor(source->GetId(), tempres.size() - 1, tempres);

        // neighbors.clear();
        // while (!tempres.empty())
        // {
        //     neighbors.emplace_back(tempres.top().GetNode());
        //     tempres.pop();
        // }
        // std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }

    void ComponentInitNSWV2::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        source->AddFriends(target, true);
        target->AddFriends(source, true);
    }

    // HNSW
    void ComponentInitHNSW::InitInner()
    {
        SetConfigs();
        Build(false);
    }

    void ComponentInitHNSW::SetConfigs()
    {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->max_m0_ = index->getParam().get<unsigned>("max_m0");

        auto ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        if (ef_construction_ > 0)
        {
            index->ef_construction_ = ef_construction_;
        }
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<int>("mult");
        // index->mult -1
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->m_));
    }

    void ComponentInitHNSW::Build(bool reverse)
    {
        // reverse False
        index->nodes_.resize(index->getBaseLen());
        int level = GetRandomNodeLevel();
        auto *first = new Index::HnswNode(0, level, index->max_m_, index->max_m0_);
        index->nodes_[0] = first;
        index->max_level_ = level;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            // 用于将接下来的循环并行化。schedule(dynamic, 128) 指示OpenMP使用动态调度，其中每个线程在完成当前分配的128个迭代后，会请求更多迭代来处理。
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                level = GetRandomNodeLevel();
                auto *qnode = new Index::HnswNode(i, level, index->max_m_, index->max_m0_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    int ComponentInitHNSW::GetRandomNodeLevel()
    {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        /*
        这行代码检查r是否小于double类型的最小正值（即接近于0），如果是，就将r设为1。这是为了避免在计算对数时出现数学错误或不稳定的行为。
        r 为1的时候 log(r) 等于0，就会插入到最底层
        index->level_mult_就是mL，作者在论文中提到 mL=1/ln(M) 是最佳选择(有实验数据证明)
        */
        return (int)(-log(r) * index->level_mult_);
    }

    void ComponentInitHNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list)
    {
        int cur_level = qnode->GetLevel();
        // 获取待插入节点 qnode 的层级
        std::unique_lock<std::mutex> max_level_lock(index->max_level_guard_, std::defer_lock);
        // 声明一个独占锁 用于在需要时锁定最大层级
        if (cur_level > index->max_level_)
            max_level_lock.lock();

        int max_level_copy = index->max_level_;
        Index::HnswNode *enterpoint = index->enterpoint_;
        // 将索引的当前最大层级复制到 max_level_copy 并获取图的入口点 enterpoint
        if (cur_level < max_level_copy)
        {
            Index::HnswNode *cur_node = enterpoint;

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                     index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                    {
                        e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + (*iter)->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());

                        s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocData() + (*iter)->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());

                        d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint = cur_node;
        }

        // PRUNE
        // std::cout << "line 268 is over" << std::endl;
        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        // std::cout << "line 270 is over" << std::endl;
        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i)
        {
            // 这个循环从最小的层级(cur_level 和 max_level_copy 之间的最小值)开始 直到达到层级0
            std::priority_queue<Index::FurtherFirst> result;
            SearchAtLayer(qnode, enterpoint, i, visited_list, result);
            a->Hnsw2Neighbor(qnode->GetId(), index->m_, result);

            while (!result.empty())
            {
                auto *top_node = result.top().GetNode();
                result.pop();
                Link(top_node, qnode, i);
                Link(qnode, top_node, i);
            }
        }

        if (cur_level > index->enterpoint_->GetLevel())
        {
            index->enterpoint_ = qnode;
            index->max_level_ = cur_level;
        }
    }

    int ComponentInitHNSW::GetRandomSeedPerThread()
    {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    void ComponentInitHNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                          Index::VisitedList *visited_list,
                                          std::priority_queue<Index::FurtherFirst> &result)
    {
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());

        float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

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

            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);

                    e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbDim());

                    s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocDim());

                    d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitHNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard()); // 使用互斥锁确保在多线程环境下对源节点 source 的访问是线程安全的
        std::vector<Index::HnswNode *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        //  获取源节点的邻居列表并添加目标节点
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        if (!shrink)
            return;

        std::priority_queue<Index::FurtherFirst> tempres;
        for (const auto &neighbor : neighbors)
        {

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + source->GetId() * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + source->GetId() * index->getBaseLocDim(),
                                                     index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            float tmp = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            tempres.push(Index::FurtherFirst(neighbor, tmp));
        }

        // PRUNE
        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        a->Hnsw2Neighbor(source->GetId(), tempres.size() - 1, tempres);

        neighbors.clear();
        while (!tempres.empty())
        {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }

    void ComponentInitGeoGraph::InitInner()
    {
        SetConfigs();
        // BuildBySkylineDescent();
        BuildByIncrementInsert();
        std::cout << "index is built over" << std::endl;
    }

    void ComponentInitGeoGraph::SetConfigs()
    {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->setInitEdgesNum(index->getParam().get<unsigned>("init_edge"));
        index->setCandidatesEdgesNum(index->getParam().get<unsigned>("candidate_edge"));
        index->setUpdateLayerNum(index->getParam().get<unsigned>("update_layer"));
        index->ITER = index->getParam().get<unsigned>("ITER");
        index->rnn_size = index->getParam().get<unsigned>("rnn_size");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->R_refine = index->getParam().get<unsigned>("R_refine");
        index->L_refine = index->getParam().get<unsigned>("L_refine");
        index->C_refine = index->getParam().get<unsigned>("C_refine");
        index->width = index->R_refine;
        index->mult = index->getParam().get<int>("mult");
        // index->mult -1
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->max_m_));
    }

    void ComponentInitGeoGraph::GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size)
    {
        unsigned N = index->getBaseLen();

        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size); // 对于addr数组的每个元素，生成一个在0到N - size之间的随机数
        }
        std::sort(addr, addr + size); // 将addr数组中的随机数进行升序排序

        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        // 遍历数组，确保每个随机数都是唯一的。如果当前随机数不大于前一个随机数，则将其设置为前一个随机数加一。这一步骤保证了即使初始随机数有重复
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
        // 生成一个0到N-1之间的随机偏移量off，然后将这个偏移量应用到数组中的每个元素上
        // 通过取模N确保结果仍在合法范围内。这样做的目的是在保持数字唯一性的同时，增加随机性
    }

    void ComponentInitGeoGraph::init()
    {
        unsigned range = index->getInitEdgesNum(); // 获取初始化边的数量

        index->geograph_nodes_.resize(index->getBaseLen()); // 根据基础长度调整最终图的大小

        index->skylinegeograph_.reserve(index->getBaseLen());

        std::mt19937 rng(rand());

#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            // 遍历基础长度
            index->geograph_nodes_[i] = new Index::GeoGraphNode(i, 0, index->max_m_);
            index->skylinegeograph_.emplace_back(Index::skyline_descent(index->getCandidatesEdgesNum(), index->getInitEdgesNum(), index->getUpdateLayerNum()));
            std::vector<unsigned> tmp(range);  // 创建临时向量存储随机生成的边
            GenRandom(rng, tmp.data(), range); // 生成随机边
            // std::vector<Index::GeoGraphSimpleNeighbor> tmp_neighbor;
            std::vector<Index::GeoGraphNNDescentNeighbor> tmp_neighbor;
            // tmp_neighbor.clear();
            // 为每个基础节点生成随机连接的节点列表
            for (unsigned j = 0; j < range; j++)
            {
                unsigned id = tmp[j]; // 获取随机生成的节点ID
                if (id == i)
                {
                    continue; // 如果随机生成的ID与当前节点相同，则跳过
                }
                float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + i * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

                float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + i * index->getBaseLocDim(),
                                                         index->getBaseLocData() + id * index->getBaseLocDim(),
                                                         index->getBaseLocDim());
                tmp_neighbor.emplace_back(Index::GeoGraphNNDescentNeighbor(id, e_d, s_d, true, -1));
            }
            sort(tmp_neighbor.begin(), tmp_neighbor.end());
            index->skylinegeograph_[i].init_neighor(tmp_neighbor);
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            auto &nnhd = index->skylinegeograph_[n];
            auto &nn_new = nnhd.nn_new;
            for (auto nn = nnhd.pool.begin(); nn != nnhd.pool.end(); ++nn)
            {
                nn_new.push_back(nn->id_);
                auto &nhood_o = index->skylinegeograph_[nn->id_]; // 根据Skyline的id得到对应的nhood
                LockGuard guard(nhood_o.lock);
                if (nhood_o.rnn_new.size() < index->rnn_size)
                    nhood_o.rnn_new.push_back(n);
                else
                {
                    unsigned int pos = rand() % index->rnn_size;
                    nhood_o.rnn_new[pos] = n;
                }
            }
        }

#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); ++i)
        {
            auto &nn_new = index->skylinegeograph_[i].nn_new;
            auto &rnn_new = index->skylinegeograph_[i].rnn_new;
            if (index->rnn_size && rnn_new.size() > index->rnn_size)
            {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(index->rnn_size);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            std::vector<unsigned>().swap(index->skylinegeograph_[i].rnn_new);
            std::vector<unsigned>().swap(index->skylinegeograph_[i].rnn_old);
        }
    }

    void ComponentInitGeoGraph::SkylineNNDescent()
    {
        auto s = std::chrono::high_resolution_clock::now();
        for (unsigned it = 0; it < index->ITER; it++)
        {
            std::cout << it << std::endl;
            join();
            update();
        }
        std::cout << "Skyline Descent is over" << std::endl;
        auto e = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        std::cout << "Initialization time for join one iter: " << duration << " milliseconds" << std::endl;
        // #ifdef PARALLEL
        // #pragma omp parallel for
        // #endif
        //         for (unsigned i = 0; i < index->getBaseLen(); i++)
        //         {
        //             Index::GeoGraphNode *source = index->geograph_nodes_[i];
        //             for (unsigned j = 0; j < index->GetinitGeoGraph()[i].size(); j++)
        //             {
        //                 int neighbor_id = index->GetinitGeoGraph()[i][j].id_;
        //                 auto *target = index->geograph_nodes_[neighbor_id];
        //                 Link(source, target);
        //             }
        //         }
    }

    void ComponentInitGeoGraph::join()
    {
        auto s = std::chrono::high_resolution_clock::now();
#ifdef PARALLEL
#pragma omp parallel for default(shared) schedule(dynamic, 100)
#endif
        for (unsigned n = 0; n < index->getBaseLen(); n++)
        {
            index->skylinegeograph_[n].join([&](unsigned i, unsigned j)
                                            {
                if (i != j) {
                    float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + i * index->getBaseEmbDim(),
                                                            index->getBaseEmbData() + j * index->getBaseEmbDim(),
                                                            index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + i * index->getBaseLocDim(),
                                                            index->getBaseLocData() + j * index->getBaseLocDim(),
                                                            index->getBaseLocDim());
                    index->skylinegeograph_[i].insert(j, e_d, s_d);
                    index->skylinegeograph_[j].insert(i, e_d, s_d);
                } });
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); n++)
        {
            auto &nn = index->skylinegeograph_[n];
            nn.updateNeighbor();
        }
        auto e = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        std::cout << "Time for Join: " << duration << " milliseconds" << std::endl;
    }

    void ComponentInitGeoGraph::update()
    {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            std::vector<unsigned>().swap(index->skylinegeograph_[i].nn_new);
            std::vector<unsigned>().swap(index->skylinegeograph_[i].nn_old);
            // 清空每个节点的新邻居和旧邻居集合
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            auto &nn = index->skylinegeograph_[n];
            unsigned maxl = std::min(nn.Q + index->getUpdateLayerNum(), (unsigned)nn.num_layer);
            unsigned c = 0;
            unsigned l = 0;
            for (unsigned i = 0; i < nn.pool.size(); i++)
            {
                if (nn.pool[i].layer_ >= maxl)
                {
                    break;
                }
                if (nn.pool[i].flag)
                {
                    ++c; // 这里更新新加入的neighbor c
                }
                l++;
            }
            nn.use_range = l;
            nn.Q += index->getUpdateLayerNum();
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            auto &nnhd = index->skylinegeograph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.use_range; ++l)
            {
                auto &nn = nnhd.pool[l]; // 根据Skyline的id得到对应的nhood
                auto &nhood_o = index->skylinegeograph_[nn.id_];
                if (nn.flag)
                {
                    nn_new.push_back(nn.id_);
                    if (nhood_o.Visited_Set.find(n) == nhood_o.Visited_Set.end())
                    {
                        // nn是n的邻居
                        // nhood_o是nn的邻居，也就是n的邻居的邻居
                        // 如果nn.distance比nhood_o.pool.back().distance, 也就是nhood_o的最大的邻居大, 那么就在nhood_o.rnn_new中加入n
                        LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_new.size() < index->rnn_size)
                            nhood_o.rnn_new.push_back(n);
                        else
                        {
                            unsigned int pos = rand() % index->rnn_size;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                }
                else
                {
                    nn_old.push_back(nn.id_);
                    if (nhood_o.Visited_Set.find(n) == nhood_o.Visited_Set.end())
                    {
                        LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_old.size() < index->rnn_size)
                            nhood_o.rnn_old.push_back(n);
                        else
                        {
                            unsigned int pos = rand() % index->rnn_size;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
        }
        float nn_new_size = 0, nn_old_size = 0;
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); ++i)
        {
            auto &nn_new = index->skylinegeograph_[i].nn_new;
            auto &nn_old = index->skylinegeograph_[i].nn_old;
            auto &rnn_new = index->skylinegeograph_[i].rnn_new;
            auto &rnn_old = index->skylinegeograph_[i].rnn_old;
            if (index->rnn_size && rnn_new.size() > index->rnn_size)
            {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(index->rnn_size);
            }
            // 如果反向新邻居列表（rnn_new）的大小超过了预设的限制（R），则将其随机打乱并裁剪到 R 的大小
            // 将处理过的反向新邻居列表添加到新邻居列表中
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (index->rnn_size && rnn_old.size() > index->rnn_size)
            {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(index->rnn_size);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if (nn_old.size() > index->rnn_size * 2)
            {
                nn_old.resize(index->rnn_size * 2);
                nn_old.reserve(index->rnn_size * 2);
            }
            nn_new_size = nn_new_size + nn_new.size();
            nn_old_size = nn_old_size + nn_old.size();
            std::vector<unsigned>().swap(index->skylinegeograph_[i].rnn_new);
            std::vector<unsigned>().swap(index->skylinegeograph_[i].rnn_old);
        }
        std::cout << "nn_new_size: " << nn_new_size / index->getBaseLen() << "\t"
                  << "nn_old_size: " << nn_old_size / index->getBaseLen() << std::endl;
    }

    void ComponentInitGeoGraph::Refine()
    {
        std::cout << "__ENTRY : GeoGraph" << std::endl;
        auto *a = new ComponentGeoGraphRefineEntryCentroid(index);
        a->EntryInner();
        std::cout << "__ENTRY : FINISH" << std::endl;
        // CANDIDATE
        std::cout << "__CANDIDATE : GeoGraph__" << std::endl;
        auto *b = new ComponentCandidateGeoGraph(index);
        // PRUNE
        std::cout << "__PRUNE : Skyline__" << std::endl;
        auto *c = new ComponentGeoGraphPruneHeuristic(index);
        // auto *cut_graph_ = new Index::GeoGraphNeighbor[index->getBaseLen() * (size_t)index->R_refine];
        std::vector<std::vector<Index::GeoGraphNeighbor>> cut_graph_;
        cut_graph_.reserve(index->getBaseLen());
        std::vector<std::mutex> locks(index->getBaseLen());
#pragma omp parallel
        {
            std::vector<Index::GeoGraphNNDescentNeighbor> pool;
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                pool.clear();
                flags.reset();
                b->CandidateInner(n, index->geograph_enterpoints, flags, pool);
                // c->PruneInner(n, index->R_refine, flags, pool, cut_graph_[n]);
            }
            std::vector<Index::GeoGraphNNDescentNeighbor>().swap(pool);

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                InterInsert(n, index->R_refine, locks, cut_graph_);
            }
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                PruneInner(n, index->R_refine, cut_graph_[n]);
            }
        }

        index->geograph_nodes_.resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++)
        {
            std::vector<Index::GeoGraphNeighbor> &pool = cut_graph_[i];
            auto &node = index->geograph_nodes_[i];
            node->SetFriends(pool);
        }
    }

    void ComponentInitGeoGraph::PruneInner(unsigned n, unsigned range,
                                           std::vector<Index::GeoGraphNeighbor> &cut_graph_)
    {
        std::vector<Index::GeoGraphNeighbor> picked;
        sort(cut_graph_.begin(), cut_graph_.end());
        std::vector<Index::GeoGraphNeighbor> candidates;
        std::vector<Index::GeoGraphNeighbor> skyline_result;
        std::vector<Index::GeoGraphNeighbor> remain_points;
        candidates.swap(cut_graph_);
        int l = 0;
        while (picked.size() < range)
        {
            findSkyline(candidates, skyline_result, remain_points);
            candidates.swap(remain_points);
            for (int i = 0; i < skyline_result.size(); i++)
            {
                std::vector<std::pair<float, float>> prune_range;
                float cur_geo_dist = skyline_result[i].geo_distance_; // s_pq
                float cur_emb_dist = skyline_result[i].emb_distance_; // e_pq
                for (size_t j = 0; j < picked.size(); j++)
                {
                    const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                    // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                    float xq_e_dist = index->get_E_Dist()->compare(
                        index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbData() + (size_t)skyline_result[i].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbDim());
                    // E(x,q)

                    float xq_s_dist = index->get_S_Dist()->compare(
                        index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                        index->getBaseLocData() + (size_t)skyline_result[i].id_ * index->getBaseLocDim(),
                        index->getBaseLocDim());
                    // S(x,q)

                    float exist_e_dist = picked[j].emb_distance_; // e_xp

                    float exist_s_dist = picked[j].geo_distance_; // s_xp

                    // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                    // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                    // if alpha holds on for the two equation at the same time, the edge will be pruned
                    // for equation 1
                    float diff1 = exist_e_dist - exist_s_dist - cur_emb_dist + cur_geo_dist; // E(p,x) - S(p,x) - E(p,q) + S(p,q)
                    float diff2 = cur_geo_dist - exist_s_dist;                               // S(p,q) - S(p,x)
                    /*
                    diff1 > 0 && diff2 > 0
                    equation 1 holds on when alpha < = diff2 / diff1
                    diff1 < 0 && diff2 < 0
                    equation 1 holds on when alpha > = diff2 / diff1
                    diff1 < 0 && diff2 > 0
                    the equation hold forever
                    diff1 > 0 && diff2 < 0
                    equation never hold which means this edge will not be pruned by this strategy
                    */
                    // float eq1_prune_upper_alpha = 1;
                    // float eq1_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_1;
                    if (diff1 > 0 && diff2 > 0)
                    {
                        // equation 1 holds on when alpha < = diff2 / diff1
                        tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                        // eq1_prune_lower_alpha = diff2 / diff1 ;
                    }
                    else if (diff1 < 0 && diff2 < 0)
                    {
                        // equation 1 holds on when alpha > = diff2 / diff1
                        // eq1_prune_upper_alpha = diff2 / diff1 ;
                        tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                    }
                    else if (diff1 < 0 && diff2 > 0)
                    {
                        tmp_prune_range_1 = {0.0f, 1.0f};
                        // the equation hold forever
                    }
                    else if (diff1 > 0 && diff2 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_1 = {0.0f, 0.0f};
                    }
                    // now for equation 2
                    float diff3 = xq_e_dist - xq_s_dist - cur_emb_dist + cur_geo_dist; //(E(q,x) - S(q,x) - E(p,q) + S(p,q))
                    float diff4 = cur_geo_dist - xq_s_dist;                            // S(p,q) - S(q,x)
                    /*
                    similar to previous
                    */
                    // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                    // float eq2_prune_upper_alpha = 1;
                    // float eq2_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_2;
                    if (diff3 > 0 && diff4 > 0)
                    {
                        // equation 2 holds on when alpha < = diff4 / diff3
                        // eq2_prune_upper_alpha = diff4 / diff3;
                        // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(0.0f, std::min(diff4 / diff3, 1.0f));
                    }
                    else if (diff3 < 0 && diff4 < 0)
                    {
                        // equation 2 holds on when alpha > = diff4 / diff3
                        // eq2_prune_lower_alpha = diff4 / diff3;
                        // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                    }
                    else if (diff3 < 0 && diff4 > 0)
                    {
                        // the equation hold forever
                        // then we do not change the previous range
                        tmp_prune_range_2 = {0.0f, 1.0f};
                    }
                    else if (diff3 > 0 && diff4 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_2 = {0.0f, 0.0f};
                    }

                    std::pair<float, float> tmp_prune_range;
                    tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                    tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                    if (tmp_prune_range.second > tmp_prune_range.first)
                    {
                        // now we consider whether this range is useful range, that is (second > first)
                        // now we check its intersection range with shared_use_range
                        intersection(picked_use_range, tmp_prune_range, prune_range);
                    }
                    else
                    {
                        continue;
                        // this range is not useful, so this edge will not be pruned by this selected edge
                    }
                }
                prune_range = mergeIntervals(prune_range);
                std::vector<std::pair<float, float>> after_pruned_use_range;
                get_use_range(prune_range, after_pruned_use_range);
                float threshold = 0.5;
                float use_size = 0;
                for (int j = 0; j < after_pruned_use_range.size(); j++)
                {
                    use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                }
                if (use_size >= threshold)
                {
                    picked.push_back(Index::GeoGraphNeighbor(skyline_result[i].id_, skyline_result[i].emb_distance_,
                                                             skyline_result[i].geo_distance_, after_pruned_use_range, l));
                }
            }
            if (picked.size() >= range || remain_points.empty())
                break;
            std::vector<Index::GeoGraphNeighbor>().swap(skyline_result);
            std::vector<Index::GeoGraphNeighbor>().swap(remain_points);
            l++;
        }
        cut_graph_.swap(picked);
    }

    void ComponentInitGeoGraph::findSkyline(std::vector<Index::GeoGraphNeighbor> &points, std::vector<Index::GeoGraphNeighbor> &skyline,
                                            std::vector<Index::GeoGraphNeighbor> &remain_points)
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

    void ComponentInitGeoGraph::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                                            std::vector<std::vector<Index::GeoGraphNeighbor>> &cut_graph_)
    {
        // Index::GeoGraphNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        std::vector<Index::GeoGraphNeighbor> src_pool = cut_graph_[n];
        for (size_t i = 0; i < src_pool.size(); i++)
        {
            if (src_pool[i].emb_distance_ == -1 && src_pool[i].geo_distance_ == -1)
                break;

            Index::GeoGraphNeighbor sn(n, src_pool[i].emb_distance_, src_pool[i].geo_distance_);
            size_t des = src_pool[i].id_;
            std::vector<Index::GeoGraphNeighbor> des_pool = cut_graph_[des];

            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < des_pool.size(); j++)
                {

                    if (n == des_pool[j].id_)
                    {
                        dup = 1;
                        break;
                    }
                }
            }
            if (dup)
                continue;
            LockGuard guard(locks[des]);
            des_pool.push_back(sn);
        }
    }

    void ComponentInitGeoGraph::BuildBySkylineDescent()
    {
        init();
        std::cout << "__INIT : Random__" << std::endl;
        SkylineNNDescent();
        std::cout << "__Refine : SkylineNNDescent__" << std::endl;
        Refine();
        std::cout << "__Refine : SkylineRefine__" << std::endl;
    }

    int ComponentInitGeoGraph::GetRandomSeedPerThread()
    {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    int ComponentInitGeoGraph::GetRandomNodeLevel()
    {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        /*
        这行代码检查r是否小于double类型的最小正值（即接近于0），如果是，就将r设为1。这是为了避免在计算对数时出现数学错误或不稳定的行为。
        r 为1的时候 log(r) 等于0，就会插入到最底层
        index->level_mult_就是mL，作者在论文中提到 mL=1/ln(M) 是最佳选择(有实验数据证明)
        */
        return (int)(-log(r) * index->level_mult_);
    }

    void ComponentInitGeoGraph::BuildByIncrementInsert()
    {
        index->geograph_nodes_.resize(index->getBaseLen());
        // int level = GetRandomNodeLevel();
        int level = 0;
        Index::GeoGraphNode *first = new Index::GeoGraphNode(0, level, index->max_m_);
        index->geograph_nodes_[0] = first;
        index->max_level_ = level;
        index->geograph_enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                std::cout << i << std::endl;
                // level = GetRandomNodeLevel();
                level = 0;
                auto *qnode = new Index::GeoGraphNode(i, level, index->max_m_);
                index->geograph_nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    void ComponentInitGeoGraph::InsertNode(Index::GeoGraphNode *qnode, Index::VisitedList *visited_list)
    {
        int cur_level = qnode->GetLevel();
        // 获取待插入节点 qnode 的层级
        std::unique_lock<std::mutex> max_level_lock(index->max_level_guard_, std::defer_lock);
        // 声明一个独占锁 用于在需要时锁定最大层级
        if (cur_level > index->max_level_)
            max_level_lock.lock();

        int max_level_copy = index->max_level_;
        Index::GeoGraphNode *enterpoint = index->geograph_enterpoint_;
        // 将索引的当前最大层级复制到 max_level_copy 并获取图的入口点 enterpoint
        std::vector<Index::GeoGraphNNDescentNeighbor> enterpoint_set;
        // std::priority_queue<Index::GeoGraphNNDescentNeighbor> enterpoint_queue;
        Index::GeoGraphNode *cur_node = enterpoint;

        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());

        enterpoint_set.emplace_back(cur_node->GetId(), e_d, s_d, true, 0);
        // the below does not work out
        // enterpoint_queue.push(Index::GeoGraphNNDescentNeighbor(cur_node->GetId(), e_d, s_d, true, 0));
        // if (cur_level < max_level_copy)
        // {
        //     for (auto i = max_level_copy; i > cur_level; --i)
        //     {
        //         // in each layer find the layer-1 skyline result as start points
        //         visited_list->Reset();
        //         for (int j = 0; j < enterpoint_set.size(); j++)
        //         {
        //             enterpoint_set[j].flag = true;
        //         }
        //         for (int j = 0; j < enterpoint_set.size();)
        //         {
        //             Index::GeoGraphNNDescentNeighbor &enterpoint_neighbor = enterpoint_set[j];
        //             if (!enterpoint_neighbor.flag)
        //             {
        //                 ++j;
        //                 continue;
        //             }
        //             enterpoint_neighbor.flag = false;
        //             visited_list->MarkAsVisited(enterpoint_neighbor.id_);
        //             cur_node = index->geograph_nodes_[enterpoint_neighbor.id_];
        //             std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
        //             const std::vector<Index::GeoGraphNeighbor> &neighbors = cur_node->GetFriends(i);
        //             for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
        //             {
        //                 if (visited_list->NotVisited(iter->id_))
        //                 {
        //                     visited_list->MarkAsVisited(iter->id_);

        //                     e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
        //                                                        index->getBaseEmbData() + iter->id_ * index->getBaseEmbDim(),
        //                                                        index->getBaseEmbDim());

        //                     s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
        //                                                        index->getBaseLocData() + iter->id_ * index->getBaseLocDim(),
        //                                                        index->getBaseLocDim());

        //                     enterpoint_set.emplace_back(iter->id_, e_d, s_d, true, 0);
        //                 }
        //             }
        //             sort(enterpoint_set.begin(), enterpoint_set.end());
        //             int pivot = -1;
        //             int nj = 0;
        //             float max_emb_dis = std::numeric_limits<float>::max();
        //             std::vector<Index::GeoGraphNNDescentNeighbor> new_enterpoint_set;
        //             while (nj < enterpoint_set.size())
        //             {
        //                 if (enterpoint_set[nj].emb_distance_ < max_emb_dis)
        //                 {
        //                     new_enterpoint_set.push_back(enterpoint_set[nj]);
        //                     max_emb_dis = enterpoint_set[nj].emb_distance_;
        //                     if (pivot == -1 && new_enterpoint_set[-1].flag == true)
        //                     {
        //                         pivot = new_enterpoint_set.size() - 1;
        //                     }
        //                 }
        //                 nj++;
        //             }
        //             enterpoint_set.swap(new_enterpoint_set);
        //             j = pivot;
        //         }
        //     }
        // }

        // PRUNE
        // for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i)
        // {
        // 这个循环从最小的层级(cur_level 和 max_level_copy 之间的最小值)开始 直到达到层级0
        std::vector<Index::GeoGraphNNDescentNeighbor> pool;
        SearchAtLayer(qnode, enterpoint_set, 0, visited_list, pool);
        ComponentGeoGraphPruneHeuristic *a = new ComponentGeoGraphPruneHeuristic(index);
        std::vector<Index::GeoGraphNeighbor> result;
        a->Geo2Neighbor(qnode->GetId(), index->max_m_, pool, result);
        for (int j = 0; j < result.size(); j++)
        {
            auto *neighbor = index->geograph_nodes_[result[j].id_];
            Link(neighbor, qnode, 0, result[j].emb_distance_, result[j].geo_distance_);
            // Link(qnode, neighbor, 0, result[j].emb_distance_, result[j].geo_distance_);
        }
        qnode->SetFriends(0, result);
        // }
        // if (cur_level > index->geograph_enterpoint_->GetLevel())
        // {
        //     index->geograph_enterpoint_ = qnode;
        //     index->max_level_ = cur_level;
        // }

        // Index::GeoGraphNode *enterpoint = index->geograph_enterpoint_;
        // std::vector<Index::GeoGraphNNDescentNeighbor> result;

        // float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
        //                                          index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
        //                                          index->getBaseEmbDim());

        // float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
        //                                          index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
        //                                          index->getBaseLocDim());
        // result.emplace_back(enterpoint->GetId(), e_d, s_d, true, -1);
        // visited_list->Reset();
        // visited_list->MarkAsVisited(enterpoint->GetId());

        // queue.init_queue(result);

        // int k = 0;
        // int l = 0;
        // while (k < queue.pool.size())
        // {
        //     while (queue.pool[k].layer_ == l)
        //     {
        //         if (queue.pool[k].flag)
        //         {
        //             queue.pool[k].flag = false;
        //             unsigned n = queue.pool[k].id_;
        //             Index::GeoGraphNode *candidate_node = index->geograph_nodes_[n];
        //             std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
        //             const std::vector<Index::GeoGraphNeighbor> &neighbors = candidate_node->GetFriends();
        //             for (unsigned m = 0; m < neighbors.size(); ++m)
        //             {
        //                 unsigned id = neighbors[m].id_;
        //                 if (visited_list->NotVisited(id))
        //                 {
        //                     visited_list->MarkAsVisited(id);

        //                     float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
        //                                                              index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
        //                                                              index->getBaseEmbDim());

        //                     float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
        //                                                              index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
        //                                                              index->getBaseLocDim());
        //                     queue.pool.emplace_back(id, e_d, s_d, true, -1);
        //                 }
        //             }
        //         }
        //         k++;
        //         if (k >= queue.pool.size())
        //         {
        //             break;
        //         }
        //     }
        //     int nk = 0;
        //     queue.updateNeighbor(nk);
        //     k = nk;
        //     if (k < queue.pool.size())
        //     {
        //         l = queue.pool[k].layer_;
        //     }
        // }
        // result.swap(queue.pool);
        // for (int i = 0; i < result.size(); i++)
        // {
        //     auto *neighbor = index->geograph_nodes_[result[i].id_];
        //     Link(neighbor, qnode, result[i].emb_distance_, result[i].geo_distance_);
        //     Link(qnode, neighbor, result[i].emb_distance_, result[i].geo_distance_);
        // }
    }

    // void ComponentInitGeoGraph::InsertNode(Index::GeoGraphNode *qnode, Index::VisitedList *visited_list)
    // {
    //     Index::GeoGraphNode *geo_enterpoint = index->geograph_enterpoint_;
    //     std::vector<float> candidate_alphas = {0.1, 0.3, 0.5, 0.7, 0.9};
    //     std::priority_queue<Index::GeoGraph_GeoCloserFirst> candidate_nodes;
    //     std::unordered_set<unsigned> visited_nodes;
    //     for (int i = 0 ; i < candidate_alphas.size(); i++)
    //     {
    //         float select_alpha = candidate_alphas[i];
    //         std::priority_queue<Index::GeoGraph_FurtherFirst> geo_result;
    //         SearchAtLayer(qnode, geo_enterpoint, visited_list, geo_result, select_alpha);
    //         while(!geo_result.empty()){
    //             const Index::GeoGraph_FurtherFirst &candidate = geo_result.top();
    //             if (visited_nodes.find(candidate.GetNode()->GetId()) == visited_nodes.end()){
    //                 candidate_nodes.emplace(Index::GeoGraph_GeoCloserFirst(candidate.GetNode(), candidate.GetEmbDistance(), candidate.GetLocDistance()));
    //                 visited_nodes.insert(candidate.GetNode()->GetId());
    //             }
    //             geo_result.pop();
    //         }
    //     }
    //     ComponentGeoGraphPruneHeuristic *a = new ComponentGeoGraphPruneHeuristic(index);
    //     std::vector<std::shared_ptr<Index::GeoGraphEdge>> selected_edge;
    //     a->Geo2Neighbor(qnode->GetId(), index->max_m_, candidate_nodes, selected_edge);
    //     for (int i = 0; i < selected_edge.size(); i++){
    //         std::shared_ptr<Index::GeoGraphEdge> neighbor = selected_edge[i];
    //         Link(neighbor->GetNode(), qnode);
    //         Link(qnode, neighbor->GetNode());
    //     }
    // }

    // void ComponentInitGeoGraph::SearchAtLayer(Index::GeoGraphNode *qnode, Index::GeoGraphNode *enterpoint,
    //                                           Index::VisitedList *visited_list,
    //                                           std::priority_queue<Index::GeoGraph_FurtherFirst> &result, float alpha)
    // {
    //     std::priority_queue<Index::GeoGraph_CloserFirst> candidates;

    //     float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
    //                                              index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
    //                                              index->getBaseEmbDim());

    //     float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
    //                                              index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
    //                                              index->getBaseLocDim());

    //     float dist = alpha * e_d + (1 - alpha) * s_d;

    //     result.emplace(enterpoint, e_d, s_d, dist);
    //     candidates.emplace(enterpoint, e_d, s_d, dist);
    //     visited_list->Reset();
    //     visited_list->MarkAsVisited(enterpoint->GetId());

    //     while (!candidates.empty())
    //     {
    //         const Index::GeoGraph_CloserFirst &candidate = candidates.top();
    //         float lower_bound = result.top().GetDistance();
    //         if (candidate.GetDistance() > lower_bound)
    //             break;
    //         Index::GeoGraphNode *candidate_node = candidate.GetNode();
    //         std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
    //         const std::vector<std::shared_ptr<Index::GeoGraphEdge>> &neighbors = candidate_node->GetFriends();
    //         candidates.pop();
    //         for (const auto &neighbor : neighbors)
    //         {
    //             Index::GeoGraphNode *neighbor_node = neighbor->GetNode();
    //             int neighbor_id = neighbor_node->GetId();
    //             const std::vector<std::pair<float, float>> &use_range = neighbor->GetRange();
    //             bool search_flag = isInRange(alpha, use_range);
    //             if (search_flag)
    //             {
    //                 if (visited_list->NotVisited(neighbor_id))
    //                 {
    //                     visited_list->MarkAsVisited(neighbor_id);

    //                     e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
    //                                                        index->getBaseEmbData() + neighbor_id * index->getBaseEmbDim(),
    //                                                        index->getBaseEmbDim());

    //                     s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
    //                                                        index->getBaseLocData() + neighbor_id * index->getBaseLocDim(),
    //                                                        index->getBaseLocDim());

    //                     dist = alpha * e_d + (1 - alpha) * s_d;

    //                     if (result.size() < index->ef_construction_ || result.top().GetDistance() > dist)
    //                     {
    //                         result.emplace(neighbor_node, e_d, s_d, dist);
    //                         candidates.emplace(neighbor_node, e_d, s_d, dist);
    //                         if (result.size() > index->ef_construction_)
    //                             result.pop();
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    void ComponentInitGeoGraph::SearchAtLayer(Index::GeoGraphNode *qnode,
                                              std::vector<Index::GeoGraphNNDescentNeighbor> &enterpoint_set,
                                              int level,
                                              Index::VisitedList *visited_list,
                                              std::vector<Index::GeoGraphNNDescentNeighbor> &result)
    {
        visited_list->Reset();
        auto queue = Index::skyline_queue(index->ef_construction_);
        for (auto &enterpoint : enterpoint_set)
        {
            result.emplace_back(enterpoint.id_, enterpoint.emb_distance_, enterpoint.geo_distance_, true, -1);
            visited_list->MarkAsVisited(enterpoint.id_);
        }
        sort(result.begin(), result.end());
        queue.init_queue(result);

        int k = 0;
        int l = 0;
        while (k < queue.pool.size())
        {
            while (queue.pool[k].layer_ == l)
            {
                if (queue.pool[k].flag)
                {
                    queue.pool[k].flag = false;
                    unsigned n = queue.pool[k].id_;
                    Index::GeoGraphNode *candidate_node = index->geograph_nodes_[n];
                    std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
                    const std::vector<Index::GeoGraphNeighbor> &neighbors = candidate_node->GetFriends(level);
                    for (unsigned m = 0; m < neighbors.size(); ++m)
                    {
                        unsigned id = neighbors[m].id_;
                        if (visited_list->NotVisited(id))
                        {
                            visited_list->MarkAsVisited(id);

                            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                                     index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                                     index->getBaseEmbDim());

                            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                                     index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                                     index->getBaseLocDim());
                            queue.pool.emplace_back(id, e_d, s_d, true, -1);
                        }
                    }
                }
                k++;
                if (k >= queue.pool.size())
                {
                    break;
                }
            }
            int nk = 0;
            queue.updateNeighbor(nk);
            k = nk;
            if (k < queue.pool.size())
            {
                l = queue.pool[k].layer_;
            }
        }
        result.swap(queue.pool);
    }

    void ComponentInitGeoGraph::Link(Index::GeoGraphNode *source, Index::GeoGraphNode *target, int level, float e_dist, float s_dist)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard());

        std::vector<Index::GeoGraphNeighbor> &neighbors = source->GetFriends(level);
        // bool shrink = neighbors.size() >= source->GetMaxM();

        // if (!shrink)
        // {
        //     std::vector<std::pair<float, float>> use_range;
        //     use_range.push_back(std::make_pair(0, 1));
        //     neighbors.emplace_back(Index::GeoGraphNeighbor(target->GetId(), e_dist, s_dist, use_range));
        //     return;
        // }
        // else
        // {
        std::vector<Index::GeoGraphNNDescentNeighbor> tempres;
        std::vector<Index::GeoGraphNeighbor> result;
        tempres.emplace_back(Index::GeoGraphNNDescentNeighbor(target->GetId(), e_dist, s_dist, true, -1));
        for (const auto &neighbor : neighbors)
        {
            tempres.emplace_back(Index::GeoGraphNNDescentNeighbor(neighbor.id_, neighbor.emb_distance_, neighbor.geo_distance_, true, -1));
        }
        neighbors.clear();
        ComponentGeoGraphPruneHeuristic *a = new ComponentGeoGraphPruneHeuristic(index);
        a->Geo2Neighbor(source->GetId(), index->max_m_, tempres, result);
        source->SetFriends(0, result);
        std::vector<Index::GeoGraphNNDescentNeighbor>().swap(tempres);
        std::vector<Index::GeoGraphNeighbor>().swap(result);
        // std::vector<Index::GeoGraphNNDescentNeighbor>().swap(tempres);
        // }
    }
}