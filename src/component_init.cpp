#include "component.h"
#include <functional>

namespace stkq
{
    void ComponentInitRTree::InitInner()
    {
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            std::cout << i << std::endl;
            double coor[2];
            coor[0] = *(index->getBaseLocData() + i * index->getBaseLocDim());
            coor[1] = *(index->getBaseLocData() + i * index->getBaseLocDim() + 1);
            index->get_R_Tree().treeInsert(coor, coor, i);
        }
    }

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

    void ComponentInitNSW::SetConfigs()
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

    void ComponentInitNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        source->AddFriends(target, true);
    }

    // baseline 4

    void ComponentInitBS4::InitInner()
    {
        SetConfigs();
        Build();
    }

    void ComponentInitBS4::SetConfigs()
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
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->m_));
    }

    int ComponentInitBS4::GetRandomNodeLevel()
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

    int ComponentInitBS4::GetRandomSeedPerThread()
    {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    void ComponentInitBS4::Build()
    {
        index->baseline4_nodes_.resize(5);
        index->baseline4_max_level_.resize(5);
        index->baseline4_enterpoint_.resize(5);

        for (size_t subindex = 0; subindex < 5; subindex++)
        {
            index->baseline4_nodes_[subindex].resize(index->getBaseLen());
        }

        int level = GetRandomNodeLevel();
        for (size_t subindex = 0; subindex < 5; subindex++)
        {
            auto *first = new Index::BS4Node(0, level, index->max_m_, index->max_m0_);
            index->baseline4_nodes_[subindex][0] = first;
            index->baseline4_enterpoint_[subindex] = first;
            index->baseline4_max_level_[subindex] = level;
        }

        for (size_t i = 1; i < index->getBaseLen(); ++i)
        {
            level = GetRandomNodeLevel();
            for (size_t subindex = 0; subindex < 5; subindex++)
            {
                // 创建节点并加入每组的 baseline4_nodes_ 中
                auto *qnode = new Index::BS4Node(i, level, index->max_m_, index->max_m0_);
                index->baseline4_nodes_[subindex][i] = qnode;
            }
        }

        for (size_t subindex = 0; subindex < 5; subindex++)
        {
            index->set_alpha(0.1 + 0.2 * subindex);
            std::cout << "Constructing alpha is " << index->get_alpha() << std::endl;
#pragma omp parallel
            {
                auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
                for (size_t i = 1; i < index->getBaseLen(); ++i)
                {
                    std::cout << i << std::endl;
                    auto *qnode = index->baseline4_nodes_[subindex][i];
                    InsertNode(qnode, visited_list, subindex);
                }
                delete visited_list;
            }
        }
    }

    void ComponentInitBS4::InsertNode(Index::BS4Node *qnode, Index::VisitedList *visited_list, unsigned subindex)
    {
        int cur_level = qnode->GetLevel();
        // 获取待插入节点 qnode 的层级
        std::unique_lock<std::mutex> max_level_lock(index->bs4_max_level_guard_, std::defer_lock);
        // 声明一个独占锁 用于在需要时锁定最大层级
        if (cur_level > index->baseline4_max_level_[subindex])
            max_level_lock.lock();

        int max_level_copy = index->baseline4_max_level_[subindex];
        Index::BS4Node *enterpoint = index->baseline4_enterpoint_[subindex];
        // 将索引的当前最大层级复制到 max_level_copy 并获取图的入口点 enterpoint
        if (cur_level < max_level_copy)
        {
            Index::BS4Node *cur_node = enterpoint;
            float e_d, s_d;
            if (index->get_alpha() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha() != 1)
            {
                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }

            float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<Index::BS4Node *> &neighbors = cur_node->GetFriends(i);

                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                    {
                        if (index->get_alpha() != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }

                        if (index->get_alpha() != 1)
                        {
                            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }

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
        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i)
        {
            // 这个循环从最小的层级(cur_level 和 max_level_copy 之间的最小值)开始 直到达到层级0
            std::priority_queue<Index::BS4FurtherFirst> result;
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

        // if (cur_level > index->baseline4_enterpoint_[subindex]->GetLevel())
        if (cur_level > index->baseline4_max_level_[subindex])
        {
            // index->enterpoint_ = qnode;
            // index->max_level_ = cur_level;
            // std::unique_lock<std::mutex> max_level_lock(index->bs4_max_level_guard_);
            index->baseline4_enterpoint_[subindex] = qnode;
            index->baseline4_max_level_[subindex] = cur_level;
        }
    }

    void ComponentInitBS4::SearchAtLayer(Index::BS4Node *qnode, Index::BS4Node *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::BS4FurtherFirst> &result)
    {
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::BS4CloserFirst> candidates;
        float e_d, s_d;

        if (index->get_alpha() != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (index->get_alpha() != 1)
        {

            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

        float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);
        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty())
        {
            const Index::BS4CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;
            Index::BS4Node *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::BS4Node *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    if (index->get_alpha() != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (index->get_alpha() != 1)
                    {
                        s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
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

    void ComponentInitBS4::Link(Index::BS4Node *source, Index::BS4Node *target, int level)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard()); // 使用互斥锁确保在多线程环境下对源节点 source 的访问是线程安全的
        std::vector<Index::BS4Node *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        //  获取源节点的邻居列表并添加目标节点
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        if (!shrink)
            return;

        std::priority_queue<Index::BS4FurtherFirst> tempres;

        float e_d, s_d;
        for (const auto &neighbor : neighbors)
        {
            if (index->get_alpha() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + source->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha() != 1)
            {
                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + source->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }

            float tmp = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;
            tempres.push(Index::BS4FurtherFirst(neighbor, tmp));
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
        std::priority_queue<Index::BS4FurtherFirst>().swap(tempres);
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
                // std::cout << i << std::endl;
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

            float e_d, s_d;
            if (index->get_alpha() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + cur_node->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha() != 1)
            {

                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + cur_node->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }

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
                        if (index->get_alpha() != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }

                        if (index->get_alpha() != 1)
                        {

                            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }
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
        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
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
        float e_d, s_d;

        if (index->get_alpha() != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (index->get_alpha() != 1)
        {

            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocData() + enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

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
                    if (index->get_alpha() != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (index->get_alpha() != 1)
                    {
                        s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
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

        float e_d, s_d;
        for (const auto &neighbor : neighbors)
        {
            if (index->get_alpha() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + source->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + neighbor->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha() != 1)
            {
                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + source->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + neighbor->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }
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

    void ComponentInitDEG::InitInner()
    {
        SetConfigs();
        BuildByIncrementInsert();
        std::cout << "index is built over" << std::endl;
    }

    void ComponentInitDEG::SetConfigs()
    {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<int>("mult");
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->max_m_));
    }

    void ComponentInitDEG::findSkyline(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &skyline,
                                       std::vector<Index::DEGNeighbor> &remain_points)
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

    void ComponentInitDEG::EntryInner()
    {
        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            index->emb_center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            {
                index->emb_center[j] += index->getBaseEmbData()[i * index->getBaseEmbDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
        {
            index->emb_center[j] /= index->getBaseLen();
        }

        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            index->loc_center[j] = 0;

        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            {
                index->loc_center[j] += index->getBaseLocData()[i * index->getBaseLocDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
        {
            index->loc_center[j] /= index->getBaseLen();
        }
    }

    void ComponentInitDEG::BuildByIncrementInsert()
    {
        index->DEG_nodes_.resize(index->getBaseLen());
        int level = 0;
        Index::DEGNode *first = new Index::DEGNode(0, index->max_m_);
        index->DEG_nodes_[0] = first;
        index->DEG_enterpoints.push_back(first);
        index->emb_center = new float[index->getBaseEmbDim()];
        index->loc_center = new float[index->getBaseLocDim()];
        EntryInner();
        index->max_level_ = level;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                std::cout << i << std::endl;
                level = 0;
                auto *qnode = new Index::DEGNode(i, index->max_m_);
                index->DEG_nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    void ComponentInitDEG::UpdateEnterpointSet(Index::DEGNode *qnode)
    {
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + qnode->GetId() * index->getBaseEmbDim(),
                                                 index->emb_center,
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + qnode->GetId() * index->getBaseLocDim(),
                                                 index->loc_center,
                                                 index->getBaseLocDim());

        {
            std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex);

            index->DEG_enterpoints_skyeline.push_back(Index::DEGNNDescentNeighbor(qnode->GetId(), e_d, s_d, true, 0));

            sort(index->DEG_enterpoints_skyeline.begin(), index->DEG_enterpoints_skyeline.end());

            float max_emb_dis = 0;
            float min_emb_dis = 1e9;

            std::vector<Index::DEGNNDescentNeighbor> skyline;
            int theta = 10;

            for (auto it = index->DEG_enterpoints_skyeline.rbegin(); it != index->DEG_enterpoints_skyeline.rend(); ++it)
            {
                if (it->emb_distance_ > max_emb_dis)
                {
                    skyline.push_back(*it);
                    max_emb_dis = it->emb_distance_;
                }
            }

            index->DEG_enterpoints_skyeline.swap(skyline);

            index->DEG_enterpoints.clear();

            for (int i = 0; i < index->DEG_enterpoints_skyeline.size(); i++)
            {
                index->DEG_enterpoints.push_back(index->DEG_nodes_[index->DEG_enterpoints_skyeline[i].id_]);
            }
        }
    }

    void ComponentInitDEG::InsertNode(Index::DEGNode *qnode, Index::VisitedList *visited_list)
    {
        std::vector<Index::DEGNNDescentNeighbor> pool;
        SearchAtLayer(qnode, visited_list, pool);
        ComponentDEGPruneHeuristic *a = new ComponentDEGPruneHeuristic(index);
        std::vector<Index::DEGNeighbor> result;
        a->DEG2Neighbor(qnode->GetId(), qnode->GetMaxM(), pool, result);
        for (int j = 0; j < result.size(); j++)
        {
            auto *neighbor = index->DEG_nodes_[result[j].id_];
            Link(neighbor, qnode, 0, result[j].emb_distance_, result[j].geo_distance_);
        }
        qnode->SetFriends(result);
        UpdateEnterpointSet(qnode);
    }

    void ComponentInitDEG::SearchAtLayer(Index::DEGNode *qnode,
                                         Index::VisitedList *visited_list,
                                         std::vector<Index::DEGNNDescentNeighbor> &pool)
    {
        visited_list->Reset();
        unsigned ef_construction = index->ef_construction_;
        unsigned query = qnode->GetId();

        std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex, std::defer_lock);

        enterpoint_lock.lock();

        for (int i = 0; i < index->DEG_enterpoints.size(); i++)
        {
            auto &enterpoint = index->DEG_enterpoints[i];

            unsigned enterpoint_id = enterpoint->GetId();

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + enterpoint_id * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + query * index->getBaseLocDim(),
                                                     index->getBaseLocData() + enterpoint_id * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            pool.emplace_back(enterpoint_id, e_d, s_d, true, 0);

            visited_list->MarkAsVisited(enterpoint_id);
        }

        enterpoint_lock.unlock();

        sort(pool.begin(), pool.end());
        auto queue = Index::skyline_queue(ef_construction);

        queue.init_queue(pool);

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
                    Index::DEGNode *candidate_node = index->DEG_nodes_[n];
                    std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
                    const std::vector<Index::DEGNeighbor> &neighbors = candidate_node->GetFriends();
                    for (unsigned m = 0; m < neighbors.size(); ++m)
                    {
                        unsigned id = neighbors[m].id_;
                        if (visited_list->NotVisited(id))
                        {
                            visited_list->MarkAsVisited(id);

                            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                                     index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                                     index->getBaseEmbDim());

                            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                                     index->getBaseLocData() + query * index->getBaseLocDim(),
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
        pool.swap(queue.pool);
    }

    void ComponentInitDEG::Link(Index::DEGNode *source, Index::DEGNode *target, int level, float e_dist, float s_dist)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard());
        std::vector<Index::DEGNeighbor> &neighbors = source->GetFriends();
        std::vector<Index::DEGNNDescentNeighbor> tempres;
        std::vector<Index::DEGNeighbor> result;
        tempres.emplace_back(Index::DEGNNDescentNeighbor(target->GetId(), e_dist, s_dist, true, -1));
        for (const auto &neighbor : neighbors)
        {
            tempres.emplace_back(Index::DEGNNDescentNeighbor(neighbor.id_, neighbor.emb_distance_, neighbor.geo_distance_, true, -1));
        }
        neighbors.clear();
        ComponentDEGPruneHeuristic *a = new ComponentDEGPruneHeuristic(index);
        a->DEG2Neighbor(source->GetId(), source->GetMaxM(), tempres, result);
        source->SetFriends(result);
        std::vector<Index::DEGNNDescentNeighbor>().swap(tempres);
        std::vector<Index::DEGNeighbor>().swap(result);
    }
}