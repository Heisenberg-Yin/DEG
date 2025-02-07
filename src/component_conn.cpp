#include "component.h"

namespace stkq
{
    // DFS
    void ComponentConnNSGDFS::ConnInner()
    {
        tree_grow();
    }

    void ComponentConnNSGDFS::tree_grow()
    {
        unsigned root = index->ep_;
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < index->getBaseLen())
        {
            DFS(flags, root, unlinked_cnt);
            // std::cout << unlinked_cnt << '\n';
            if (unlinked_cnt >= index->getBaseLen())
                break;
            findroot(flags, root);
            // std::cout << "new root"<<":"<<root << '\n';
        }
        for (size_t i = 0; i < index->getBaseLen(); ++i)
        {
            if (index->getFinalGraph()[i].size() > index->width)
            {
                index->width = index->getFinalGraph()[i].size();
            }
        }
    }

    void ComponentConnNSGDFS::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt)
    {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root])
            cnt++;
        flag[root] = true;
        while (!s.empty())
        {
            unsigned next = index->getBaseLen() + 1;
            for (unsigned i = 0; i < index->getFinalGraph()[tmp].size(); i++)
            {
                if (!flag[index->getFinalGraph()[tmp][i].id])
                {
                    next = index->getFinalGraph()[tmp][i].id;
                    break;
                }
            }
            // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
            if (next == (index->getBaseLen() + 1))
            {
                s.pop();
                if (s.empty())
                    break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }

    void ComponentConnNSGDFS::findroot(boost::dynamic_bitset<> &flag, unsigned &root)
    {
        unsigned id = index->getBaseLen();
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            if (flag[i] == false)
            {
                id = i;
                break;
            }
        }

        if (id == index->getBaseLen())
            return; // No Unlinked Node

        std::vector<Index::Neighbor> tmp, pool;
        // get_neighbors(index->getBaseData() + index->getBaseDim() * id, tmp, pool);
        get_neighbors(id, tmp, pool);
        std::sort(pool.begin(), pool.end());

        unsigned found = 0;
        for (unsigned i = 0; i < pool.size(); i++)
        {
            if (flag[pool[i].id])
            {
                // std::cout << pool[i].id << '\n';
                root = pool[i].id;

                found = 1;
                break;
            }
        }
        if (found == 0)
        {
            while (true)
            {
                unsigned rid = rand() % index->getBaseLen();
                if (flag[rid])
                {
                    root = rid;
                    break;
                }
            }
        }
        // float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * root,
        //                                  index->getBaseData() + index->getBaseDim() * id,
        //                                  index->getBaseDim());
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + root * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + root * index->getBaseLocDim(),
                                                 index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                 index->getBaseLocDim());

        float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

        index->getFinalGraph()[root].push_back(Index::SimpleNeighbor(id, dist));
    }

    void ComponentConnNSGDFS::get_neighbors(const int query, std::vector<Index::Neighbor> &retset,
                                            std::vector<Index::Neighbor> &fullset)
    {
        unsigned L = index->getParam().get<unsigned>("L_refine");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[index->ep_].size(); i++)
        {
            init_ids[i] = index->getFinalGraph()[index->ep_][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size())
        {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen())
                continue;
            // std::cout<<id<<std::endl;
            // float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t) id, query,
            //                                        (unsigned) index->getBaseDim());

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + id * index->getBaseLocDim(),
                                                     index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            retset[i] = Index::Neighbor(id, dist, true);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;

            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m)
                {
                    unsigned id = index->getFinalGraph()[n][m].id;
                    if (flags[id])
                        continue;
                    flags[id] = 1;

                    // float dist = index->getDist()->compare(query,
                    //                                        index->getBaseData() + index->getBaseDim() * (size_t) id,
                    //                                        (unsigned) index->getBaseDim());
                    float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                             index->getBaseEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                             index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + id * index->getBaseLocDim(),
                                                             index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
                                                             index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    Index::Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance)
                        continue;
                    int r = Index::InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size())
                        ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }
}