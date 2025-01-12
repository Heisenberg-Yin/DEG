#include "component.h"

namespace stkq
{

    void ComponentRefineEntryCentroid::EntryInner()
    {
        auto *emb_center = new float[index->getBaseEmbDim()];
        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            emb_center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            {
                emb_center[j] += index->getBaseEmbData()[i * index->getBaseEmbDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
        {
            emb_center[j] /= index->getBaseLen();
        }

        auto *loc_center = new float[index->getBaseLocDim()];

        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            loc_center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            {
                loc_center[j] += index->getBaseLocData()[i * index->getBaseLocDim() + j];
            }
        }

        std::vector<Index::Neighbor> tmp, pool;
        index->ep_ = rand() % index->getBaseLen(); // random initialize navigating point
        get_neighbors(emb_center, loc_center, tmp, pool);
        index->ep_ = tmp[0].id;
        std::cout << "ep_ " << index->ep_ << std::endl;
    }

    void ComponentRefineEntryCentroid::get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::Neighbor> &retset,
                                                     std::vector<Index::Neighbor> &fullset)
    {
        unsigned L = index->L_refine;
        // 从index的参数中获取L_refine值
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[index->ep_].size(); i++)
        {
            init_ids[i] = index->getFinalGraph()[index->ep_][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        // 使用当前导航点的邻接点作为初始候选点

        while (L < init_ids.size())
        {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        // 如果候选点不足，通过随机选择来补充
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen())
                continue;
            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                     query_emb,
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                     query_loc,
                                                     index->getBaseLocDim());

            float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            retset[i] = Index::Neighbor(id, dist, true);
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        // 计算距离并更新近邻集合
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (retset[k].flag)
            {
                retset[k].flag = false;
                // 表示该点已经处理过
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m)
                {
                    unsigned id = index->getFinalGraph()[n][m].id;
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                             query_emb,
                                                             index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                             query_loc,
                                                             index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    Index::Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance)
                        continue;
                    int r = Index::InsertIntoPool(retset.data(), L, nn);
                    // 如果计算得到的距离小于结果集中最远点的距离，则尝试将其插入到结果集 retset 中
                    if (L + 1 < retset.size())
                    {
                        ++L;
                    }
                    if (r < nk)
                    {
                        nk = r;
                        // 如果新近邻插入位置 nk 小于等于当前索引 k，则更新 k 为 nk，以便重新从新的插入点开始遍历
                    }
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void ComponentDEGRefineEntryCentroid::EntryInner()
    {
        auto *emb_center = new float[index->getBaseEmbDim()];
        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            emb_center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            {
                emb_center[j] += index->getBaseEmbData()[i * index->getBaseEmbDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
        {
            emb_center[j] /= index->getBaseLen();
        }

        auto *loc_center = new float[index->getBaseLocDim()];

        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            loc_center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            {
                loc_center[j] += index->getBaseLocData()[i * index->getBaseLocDim() + j];
            }
        }
        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
        {
            loc_center[j] /= index->getBaseLen();
        }

        std::vector<Index::DEGNNDescentNeighbor> tmp, pool;
        index->ep_ = rand() % index->getBaseLen(); // random initialize navigating point
        get_neighbors(emb_center, loc_center, tmp, pool);
        // for (int i = 0; i < tmp.size(); i++)
        // {
        //     if (tmp[i].layer_ == 0)
        //         index->DEG_enterpoints.push_back(tmp[i].id_);
        //     else
        //         break;
        // }
    }

    void ComponentDEGRefineEntryCentroid::get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::DEGNNDescentNeighbor> &retset,
                                                             std::vector<Index::DEGNNDescentNeighbor> &fullset)
    {
        unsigned L = index->L_refine;
        // 从index的参数中获取L_refine值
        retset.reserve(L);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getSkylineGraph()[index->ep_].pool.size(); i++)
        {
            init_ids[i] = index->getSkylineGraph()[index->ep_].pool[i].id_;
            flags[init_ids[i]] = true;
            L++;
        }
        // 使用当前导航点的邻接点作为初始候选点

        while (L < init_ids.size())
        {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        // 如果候选点不足，通过随机选择来补充
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen())
                continue;
            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                     query_emb,
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                     query_loc,
                                                     index->getBaseLocDim());

            retset.emplace_back(id, e_d, s_d, true, -1);
            L++;
        }
        std::sort(retset.begin(), retset.end());
        // 计算距离并更新近邻集合
        auto queue = Index::skyline_queue(L);
        queue.init_queue(retset);
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
                    for (unsigned m = 0; m < index->getSkylineGraph()[n].pool.size(); ++m)
                    {
                        unsigned id = index->getSkylineGraph()[n].pool[m].id_;
                        if (flags[id])
                            continue;
                        flags[id] = true;
                        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                                 query_emb,
                                                                 index->getBaseEmbDim());

                        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                                 query_loc,
                                                                 index->getBaseLocDim());
                        queue.pool.emplace_back(id, e_d, s_d, true, -1);
                    }
                }
                k++;
            }
            // int nk = k;
            // int nl = l;
            // queue.updateNeighbor(nk);
            // if (nk < k)
            // {
            //     k = nk;
            //     l = nl;
            // }
            // else
            // {
            //     l++;
            // }
        }
        retset.swap(queue.pool);
        std::vector<unsigned>().swap(init_ids);
    }
}