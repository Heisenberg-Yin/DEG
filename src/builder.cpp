
#include "builder.h"
#include "component.h"
#include "rtree.h"
#include <set>

namespace stkq
{
    /**
     * load dataset and parameters
     * param data_emb_file *base_emb_norm.fvecs
     * param data_emb_file *base_loc_norm.fvecs
     * param query_emb_file *_query_emb_norm.fvecs
     * param query_loc_file *_query_loc_norm.fvecs
     * param ground_file *_groundtruth.ivecs
     * param parameters
     * return pointer of builder
     */

    IndexBuilder *IndexBuilder::load(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *ground_file, Parameters &parameters, bool dual)
    {
        if (!dual)
        {
            auto *a = new ComponentLoad(final_index_);
            a->LoadInner(data_emb_file, data_loc_file, query_emb_file, query_loc_file, ground_file, parameters);
            std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
            std::cout << "base data emb dim : " << final_index_->getBaseEmbDim() << std::endl;
            std::cout << "base data loc dim : " << final_index_->getBaseLocDim() << std::endl;
            std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
            std::cout << "query data emb dim : " << final_index_->getQueryEmbDim() << std::endl;
            std::cout << "query data loc dim : " << final_index_->getQueryLocDim() << std::endl;
            std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
            std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << final_index_->getParam().toString() << std::endl;
            std::cout << "=====================" << std::endl;
            return this;
        }
        else
        {
            auto *a = new ComponentLoad(final_index_1);
            a->LoadInner(data_emb_file, data_loc_file, query_emb_file, query_loc_file, ground_file, parameters);
            final_index_1->set_alpha(0);
            auto *b = new ComponentLoad(final_index_2);
            b->LoadInner(data_emb_file, data_loc_file, query_emb_file, query_loc_file, ground_file, parameters);
            final_index_2->set_alpha(1);
            std::cout << "base data len : " << final_index_1->getBaseLen() << std::endl;
            std::cout << "base data emb dim : " << final_index_1->getBaseEmbDim() << std::endl;
            std::cout << "base data loc dim : " << final_index_1->getBaseLocDim() << std::endl;
            std::cout << "query data len : " << final_index_1->getQueryLen() << std::endl;
            std::cout << "query data emb dim : " << final_index_1->getQueryEmbDim() << std::endl;
            std::cout << "query data loc dim : " << final_index_1->getQueryLocDim() << std::endl;
            std::cout << "ground truth data len : " << final_index_1->getGroundLen() << std::endl;
            std::cout << "ground truth data dim : " << final_index_1->getGroundDim() << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << final_index_1->getParam().toString() << std::endl;
            std::cout << final_index_2->getParam().toString() << std::endl;
            std::cout << final_index_1->get_alpha() << std::endl;
            std::cout << final_index_2->get_alpha() << std::endl;
            std::cout << "=====================" << std::endl;
            return this;
        }
    }

    /**
     * build init graph
     * param type init type
     * param debug Whether to output graph index information (will have a certain impact on performance)
     * return pointer of builder
     */

    IndexBuilder *IndexBuilder::init(TYPE type, bool debug)
    {
        s = std::chrono::high_resolution_clock::now();
        ComponentInit *a = nullptr;

        if (type == INIT_HNSW)
        {
            std::cout << "__INIT : HNSW__" << std::endl;
            a = new ComponentInitHNSW(final_index_);
        }
        else if (type == INIT_DEG)
        {
            std::cout << "__INIT : GEO_RNG__" << std::endl;
            a = new ComponentInitDEG(final_index_);
        }
        else if (type == INIT_RTREE)
        {
            std::cout << "__INIT : RTREE__" << std::endl;
            a = new ComponentInitRTree(final_index_);
        }
        else if (type == INIT_BS4)
        {
            std::cout << "__INIT : BASELINE_4__" << std::endl;
            a = new ComponentInitBS4(final_index_);
        }
        else
        {
            std::cerr << "__INIT : WRONG TYPE__" << std::endl;
            exit(-1);
        }
        a->InitInner();
        e = std::chrono::high_resolution_clock::now();
        std::cout << "__INIT FINISH__" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        std::cout << "Initialization time: " << duration << " milliseconds" << std::endl;
        return this;
    }

    /**
     * build refine graph
     * @param type refine type
     * @param debug
     * @return
     */
    IndexBuilder *IndexBuilder::refine(TYPE type, bool debug)
    {
        ComponentRefine *a = nullptr;

        if (type == REFINE_NN_DESCENT)
        {
            std::cout << "__REFINE : NNDscent" << std::endl;
            a = new ComponentRefineNNDescent(final_index_);
        }
        else if (type == REFINE_NSG)
        {
            std::cout << "__REFINE : NSG__" << std::endl;
            a = new ComponentRefineNSG(final_index_);
        }
        else
        {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->RefineInner();

        std::cout << "===================" << std::endl;
        std::cout << "__REFINE : FINISH__" << std::endl;
        std::cout << "===================" << std::endl;
        e = std::chrono::high_resolution_clock::now();
        return this;
    }

    IndexBuilder *IndexBuilder::save_graph(TYPE type, char *graph_file)
    {

        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        if (type == INDEX_NSG)
        {
            out.write((char *)&final_index_->ep_, sizeof(unsigned));
        }
        else if (type == INDEX_HNSW)
        {
            unsigned enterpoint_id = final_index_->enterpoint_->GetId();
            unsigned max_level = final_index_->max_level_;
            out.write((char *)&enterpoint_id, sizeof(unsigned));
            out.write((char *)&max_level, sizeof(unsigned));
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned node_level = final_index_->nodes_[i]->GetLevel() + 1;
                out.write((char *)&node_level, sizeof(unsigned));
                unsigned current_level_GK;
                for (unsigned j = 0; j < node_level; j++)
                {
                    current_level_GK = final_index_->nodes_[i]->GetFriends(j).size();
                    out.write((char *)&current_level_GK, sizeof(unsigned));
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id = final_index_->nodes_[i]->GetFriends(j)[k]->GetId();
                        out.write((char *)&current_level_neighbor_id, sizeof(unsigned));
                    }
                }
            }
            out.close();
            return this;
        }
        else if (type == INDEX_NSW)
        {
            for (unsigned i = 0; i < final_index_->nodes_.size(); i++)
            {
                unsigned GK = (unsigned)final_index_->nodes_[i]->GetFriends(0).size();
                unsigned node_id = final_index_->nodes_[i]->GetId();
                std::vector<unsigned> tmp;
                for (unsigned j = 0; j < GK; j++)
                {
                    tmp.push_back((unsigned)final_index_->nodes_[i]->GetFriends(0)[j]->GetId());
                }
                out.write((char *)&node_id, sizeof(unsigned));
                out.write((char *)&GK, sizeof(unsigned));
                out.write((char *)tmp.data(), GK * sizeof(unsigned));
            }
            out.close();
            return this;
        }
        else if (type == INDEX_DEG)
        {
            int average_neighbor_size = 0;
            unsigned enterpoint_set_size = final_index_->DEG_enterpoints.size();
            out.write((char *)&enterpoint_set_size, sizeof(unsigned));
            for (unsigned i = 0; i < enterpoint_set_size; i++)
            {
                unsigned node_id = final_index_->DEG_enterpoints[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
            }

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->DEG_nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetFriends().size();
                out.write((char *)&neighbor_size, sizeof(unsigned));
                average_neighbor_size = average_neighbor_size + neighbor_size;

                for (unsigned k = 0; k < neighbor_size; k++)
                {
                    Index::DEGNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetFriends()[k];
                    unsigned neighbor_id = neighbor.id_;
                    out.write((char *)&neighbor_id, sizeof(unsigned));
                    std::vector<std::pair<float, float>> use_range = neighbor.available_range;

                    // unsigned range_size = use_range.size();
                    // out.write((char *)&range_size, sizeof(unsigned));
                    // for (unsigned t = 0; t < range_size; t++)
                    // {
                    //     out.write((char *)&use_range[t].first, sizeof(float));
                    //     out.write((char *)&use_range[t].second, sizeof(float));
                    // }

                    unsigned range_size = use_range.size();
                    out.write((char *)&range_size, sizeof(unsigned));
                    for (unsigned t = 0; t < range_size; t++)
                    {
                        int8_t x = static_cast<int8_t>(use_range[t].first * 100);
                        int8_t y = static_cast<int8_t>(use_range[t].second * 100);                        
                        out.write((char *)x, sizeof(x));
                        out.write((char *)y, sizeof(y));
                    }
                }
            }
            out.close();
            return this;
        }
        else if (type == INDEX_RTREE)
        {
            out.close();

            if (final_index_->get_R_Tree().saveIndex(graph_file))
            {
                std::cout << "succuessful save " << graph_file << std::endl;
            }
            else
            {
                std::cout << "error for saving" << std::endl;
            }
            return this;
        }

        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned GK = (unsigned)final_index_->getFinalGraph()[i].size();
            std::vector<unsigned> tmp;
            for (unsigned j = 0; j < GK; j++)
            {
                tmp.push_back(final_index_->getFinalGraph()[i][j].id);
            }
            out.write((char *)&GK, sizeof(unsigned));
            out.write((char *)tmp.data(), GK * sizeof(unsigned));
        }
        out.close();

        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file)
    {
        std::ifstream in(graph_file, std::ios::binary);
        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        if (!in.is_open())
        {
            std::cerr << "load graph error: " << graph_file << std::endl;
            exit(-1);
        }
        if (type == INDEX_HNSW)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in.read((char *)&enterpoint_id, sizeof(unsigned));
            in.read((char *)&final_index_->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in.read((char *)&node_id, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                in.read((char *)&node_level, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_->nodes_[current_level_neighbor_id]);
                    }
                    final_index_->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;

            final_index_->enterpoint_ = final_index_->nodes_[enterpoint_id];
            return this;
        }
        else if (type == INDEX_NSW)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            while (!in.eof())
            {
                unsigned GK, node_id;
                in.read((char *)&node_id, sizeof(unsigned));
                in.read((char *)&GK, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                if (in.eof())
                    break;
                std::vector<unsigned> tmp(GK);
                in.read((char *)tmp.data(), GK * sizeof(unsigned));
                for (unsigned j = 0; j < tmp.size(); j++)
                {
                    final_index_->nodes_[tmp[j]]->SetId(tmp[j]);
                    final_index_->nodes_[node_id]->AddFriends(final_index_->nodes_[tmp[j]], false);
                }
            }
            return this;
        }
        else if (type == INDEX_NSWV2)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            while (!in.eof())
            {
                unsigned GK, node_id;
                in.read((char *)&node_id, sizeof(unsigned));
                in.read((char *)&GK, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                if (in.eof())
                    break;
                std::vector<unsigned> tmp(GK);
                in.read((char *)tmp.data(), GK * sizeof(unsigned));
                for (unsigned j = 0; j < tmp.size(); j++)
                {
                    final_index_->nodes_[tmp[j]]->SetId(tmp[j]);
                    final_index_->nodes_[node_id]->AddFriends(final_index_->nodes_[tmp[j]], false);
                }
            }
            return this;
        }
        else if (type == INDEX_NSG)
        {
            in.read((char *)&final_index_->ep_, sizeof(unsigned));
        }
        else if (type == INDEX_DEG)
        {
            int average_neighbor_size = 0;
            final_index_->DEG_nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->DEG_nodes_[i] = new stkq::DEG::DEGNode(0, 0, 0);
            }
            unsigned enterpoint_id, enterpoint_size;
            final_index_->enterpoint_set.clear();
            in.read((char *)&enterpoint_size, sizeof(unsigned));
            for (unsigned i = 0; i < enterpoint_size; i++)
            {
                in.read((char *)&enterpoint_id, sizeof(unsigned));
                final_index_->enterpoint_set.push_back(enterpoint_id);
            }
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, neighbor_size;
                in.read((char *)&node_id, sizeof(unsigned));
                final_index_->DEG_nodes_[i]->SetId(node_id);
                in.read((char *)&neighbor_size, sizeof(unsigned));
                average_neighbor_size = average_neighbor_size + neighbor_size;
                // if (j == 0)
                // {
                //     average_neighbor_size = average_neighbor_size + neighbor_size;
                // }
                final_index_->DEG_nodes_[i]->SetMaxM(neighbor_size);
                std::vector<Index::DEGSimpleNeighbor> neighbors;
                neighbors.reserve(neighbor_size);
                int max_layer = 0;
                for (unsigned k = 0; k < neighbor_size; k++)
                {
                    unsigned neighbor_id;
                    in.read((char *)&neighbor_id, sizeof(unsigned));
                    unsigned range_size;
                    in.read((char *)&range_size, sizeof(unsigned));
                                        
                    std::vector<std::pair<float, float>> use_range;
                    for (unsigned t = 0; t < range_size; t++)
                    {
                        float range_start, range_end;
                        in.read((char *)&range_start, sizeof(float));
                        in.read((char *)&range_end, sizeof(float));
                        use_range.push_back(std::make_pair(range_start, range_end));
                    }
                    // neighbors.push_back(std::make_shared<Index::DEGEdge>(final_index_->DEG_nodes_[neighbor_id], e_dist, s_dist, use_range));
                    neighbors.push_back(Index::DEGSimpleNeighbor(neighbor_id, use_range));
                }
                final_index_->DEG_nodes_[i]->SetSearchFriends(neighbors);
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;
            return this;
        }

        while (!in.eof())
        {
            unsigned GK;
            in.read((char *)&GK, sizeof(unsigned));
            if (in.eof())
                break;
            std::vector<unsigned> tmp(GK);
            in.read((char *)tmp.data(), GK * sizeof(unsigned));
            final_index_->getLoadGraph().push_back(tmp);
        }
        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file_1, char *graph_file_2)
    {
        std::ifstream in1(graph_file_1, std::ios::binary);
        std::ifstream in2(graph_file_2, std::ios::binary);

        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        if (!in1.is_open())
        {
            std::cerr << "load graph error: " << graph_file_1 << std::endl;
            exit(-1);
        }

        if (!in2.is_open())
        {
            std::cerr << "load graph error: " << graph_file_2 << std::endl;
            exit(-1);
        }

        if (type == INDEX_HNSW)
        {
            final_index_1->nodes_.resize(final_index_1->getBaseLen());
            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                final_index_1->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in1.read((char *)&enterpoint_id, sizeof(unsigned));
            in1.read((char *)&final_index_1->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in1.read((char *)&node_id, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetId(node_id);
                in1.read((char *)&node_level, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in1.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in1.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_1->nodes_[current_level_neighbor_id]);
                    }
                    final_index_1->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_1->getBaseLen() << std::endl;
            final_index_1->enterpoint_ = final_index_1->nodes_[enterpoint_id];

            average_neighbor_size = 0;
            l1_average_neighbor_size = 0;
            final_index_2->nodes_.resize(final_index_2->getBaseLen());
            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                final_index_2->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            in2.read((char *)&enterpoint_id, sizeof(unsigned));
            in2.read((char *)&final_index_2->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in2.read((char *)&node_id, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetId(node_id);
                in2.read((char *)&node_level, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in2.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in2.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_2->nodes_[current_level_neighbor_id]);
                    }
                    final_index_2->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_2->getBaseLen() << std::endl;
            final_index_2->enterpoint_ = final_index_2->nodes_[enterpoint_id];
        }
        else if (type == INDEX_RTREE_HNSW)
        {
            final_index_1->get_R_Tree().loadIndex(graph_file_1);

            unsigned enterpoint_id;
            average_neighbor_size = 0;
            l1_average_neighbor_size = 0;
            final_index_2->nodes_.resize(final_index_2->getBaseLen());
            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                final_index_2->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            in2.read((char *)&enterpoint_id, sizeof(unsigned));
            in2.read((char *)&final_index_2->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in2.read((char *)&node_id, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetId(node_id);
                in2.read((char *)&node_level, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in2.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in2.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_2->nodes_[current_level_neighbor_id]);
                    }
                    final_index_2->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_2->getBaseLen() << std::endl;
            final_index_2->enterpoint_ = final_index_2->nodes_[enterpoint_id];
        }
        else
        {
            std::cout << "error for index type" << std::endl;
            exit(1);
        }

        return this;
    }

    /**
     * offline search
     * param entry_type
     * param route_type
     * return
     */
    IndexBuilder *IndexBuilder::search(TYPE entry_type, TYPE route_type, TYPE L_type, Parameters param_)
    {
        std::cout << "__SEARCH__" << std::endl;

        unsigned K = 10; // 在近邻搜索中要找到的最近邻的数量

        if (route_type == DUAL_ROUTER_HNSW)
        {
            final_index_1->getParam().set<unsigned>("K_search", K);
            final_index_2->getParam().set<unsigned>("K_search", K);
            std::vector<std::vector<unsigned>> res_1;
            std::vector<std::vector<unsigned>> res_2;
            std::cout << "__ROUTER : DUAL_HNSW__" << std::endl;
            ComponentSearchEntry *a1 = new ComponentSearchEntryNone(final_index_1);
            ComponentSearchEntry *a2 = new ComponentSearchEntryNone(final_index_2);
            ComponentSearchRoute *b1 = new ComponentSearchRouteHNSW(final_index_1);
            ComponentSearchRoute *b2 = new ComponentSearchRouteHNSW(final_index_2);
            if (L_type == L_SEARCH_ASCEND)
            {
                std::set<unsigned> visited;
                unsigned sg = 1000;
                float acc_set = 0.99;
                bool flag = false;
                int L_sl = 1;
                unsigned L = 0;
                unsigned k_plus = 0;
                visited.insert(L);
                unsigned L_min = 0x7fffffff;
                float alpha = param_.get<float>("alpha");
                for (unsigned t = 0; t < 20; t++)
                {
                    L = L + K;
                    final_index_1->getParam().set<unsigned>("K_search", L);
                    final_index_2->getParam().set<unsigned>("K_search", L);
                    std::cout << "SEARCH_L : " << L << std::endl;
                    if (L < K)
                    {
                        std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                        exit(-1);
                    }

                    final_index_1->getParam().set<unsigned>("L_search", L);
                    final_index_2->getParam().set<unsigned>("L_search", L);

                    auto s1 = std::chrono::high_resolution_clock::now();

                    res_1.clear();
                    res_1.resize(final_index_1->getQueryLen());
                    for (unsigned i = 0; i < final_index_1->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a1->SearchEntryInner(i, pool);
                        b1->RouteInner(i, pool, res_1[i]);
                    }

                    res_2.clear();
                    res_2.resize(final_index_2->getQueryLen());
                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a2->SearchEntryInner(i, pool);
                        b2->RouteInner(i, pool, res_2[i]);
                    }

                    std::priority_queue<Index::CloserFirst> result_queue;
                    std::vector<std::vector<unsigned>> res;

                    for (int i = 0; i < res_1.size(); i++)
                    {
                        for (int j = 0; j < res_1[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_1[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_1[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float d = alpha * e_d + (1 - alpha) * s_d;

                            result_queue.emplace(final_index_1->nodes_[res_1[i][j]], d);
                        }

                        for (int j = 0; j < res_2[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_2[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_2[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float d = alpha * e_d + (1 - alpha) * s_d;

                            result_queue.emplace(final_index_1->nodes_[res_2[i][j]], d);
                        }
                        std::vector<unsigned> tmp_res;
                        // std::unordered_set<unsigned> unique_results;
                        while (!result_queue.empty())
                        {
                            int top_node_id = result_queue.top().GetNode()->GetId();
                            // if (tmp_res.size() < K && unique_results.find(top_node_id) == unique_results.end())
                            if (tmp_res.size() < K)
                            {
                                tmp_res.push_back(top_node_id);
                                // unique_results.insert(top_node_id);
                            }
                            result_queue.pop();
                        }
                        res.push_back(tmp_res);
                    }

                    auto e1 = std::chrono::high_resolution_clock::now();
                    // auto e2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = e1 - s1;
                    std::cout << "search time: " << diff.count() / final_index_1->getQueryLen() << "\n";

                    float recall = 0;

                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        // if (res_1[i].size() == 0 or res_2[i].size() == 0)
                        if (res[i].size() == 0)
                            continue;
                        float tmp_recall = 0;
                        float cnt = 0;

                        for (unsigned j = 0; j < K; j++)
                        {
                            unsigned k = 0;
                            for (; k < K; k++)
                            {
                                if (res[i][k] == final_index_2->getGroundData()[i * final_index_2->getGroundDim() + j])
                                    break;
                            }
                            if (k == K)
                                cnt++;
                        }
                        tmp_recall = (float)(K - cnt) / (float)K;
                        recall = recall + tmp_recall;
                    }
                    // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                    float acc = recall / final_index_2->getQueryLen();
                    std::cout << K << " NN accuracy: " << acc << std::endl;
                }
            }
            e = std::chrono::high_resolution_clock::now();
            std::cout << "__SEARCH FINISH__" << std::endl;

            return this;
        }
        else if (route_type == ROUTER_RTREE_HNSW)
        {
            // final_index_1->getParam().set<unsigned>("K_search", K);
            final_index_2->getParam().set<unsigned>("K_search", K);
            std::vector<std::vector<unsigned>> res_1;
            std::vector<std::vector<unsigned>> res_2;
            std::cout << "__ROUTER : ROUTER_RTREE_HNSW__" << std::endl;
            ComponentSearchEntry *a2 = new ComponentSearchEntryNone(final_index_2);
            ComponentSearchRoute *b2 = new ComponentSearchRouteHNSW(final_index_2);
            if (L_type == L_SEARCH_ASCEND)
            {
                std::set<unsigned> visited;
                unsigned sg = 1000;
                float acc_set = 0.99;
                bool flag = false;
                int L_sl = 1;
                unsigned L = 0;
                unsigned k_plus = 0;
                visited.insert(L);
                unsigned L_min = 0x7fffffff;
                float alpha = param_.get<float>("alpha");
                auto &rtree = final_index_1->get_R_Tree();

                for (unsigned t = 0; t < 30; t++)
                {
                    L = L + K;
                    final_index_2->getParam().set<unsigned>("K_search", L);
                    std::cout << "SEARCH_L : " << L << std::endl;
                    if (L < K)
                    {
                        std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                        exit(-1);
                    }
                    final_index_2->getParam().set<unsigned>("L_search", L);

                    auto s1 = std::chrono::high_resolution_clock::now();

                    res_1.clear();
                    res_1.resize(final_index_1->getQueryLen());

                    for (unsigned i = 0; i < final_index_1->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        Point q = std::make_pair(
                            *(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim()),
                            *(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim() + 1));
                        rtree.query(q, L, final_index_1->getBaseLocData(), res_1[i]);
                    }

                    auto s2 = std::chrono::high_resolution_clock::now();

                    res_2.clear();
                    res_2.resize(final_index_2->getQueryLen());
                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a2->SearchEntryInner(i, pool);
                        b2->RouteInner(i, pool, res_2[i]);
                    }
                    std::cout << "DistCount: " << final_index_2->getDistCount() << std::endl;
                    std::cout << "HopCount: " << final_index_2->getHopCount() << std::endl;
                    final_index_2->resetDistCount();
                    final_index_2->resetHopCount();
                    // int cnt = 0;
                    std::priority_queue<Index::CloserFirst> result_queue;
                    std::vector<std::vector<unsigned>> res;

                    for (int i = 0; i < res_1.size(); i++)
                    {
                        for (int j = 0; j < res_1[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_1[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_1[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float d = alpha * e_d + (1 - alpha) * s_d;

                            result_queue.emplace(final_index_2->nodes_[res_1[i][j]], d);
                        }

                        for (int j = 0; j < res_2[i].size(); j++)
                        {
                            float e_d = final_index_2->get_E_Dist()->compare(final_index_2->getQueryEmbData() + i * final_index_2->getBaseEmbDim(),
                                                                             final_index_2->getBaseEmbData() + res_2[i][j] * final_index_2->getBaseEmbDim(),
                                                                             final_index_2->getBaseEmbDim());

                            float s_d = final_index_2->get_S_Dist()->compare(final_index_2->getQueryLocData() + i * final_index_2->getBaseLocDim(),
                                                                             final_index_2->getBaseLocData() + res_2[i][j] * final_index_2->getBaseLocDim(),
                                                                             final_index_2->getBaseLocDim());

                            float d = alpha * e_d + (1 - alpha) * s_d;

                            result_queue.emplace(final_index_2->nodes_[res_2[i][j]], d);
                        }
                        std::vector<unsigned> tmp_res;

                        while (!result_queue.empty())
                        {
                            if (tmp_res.size() < K)
                            {
                                tmp_res.push_back(result_queue.top().GetNode()->GetId());
                            }
                            result_queue.pop();
                        }
                        res.push_back(tmp_res);
                    }

                    auto e1 = std::chrono::high_resolution_clock::now();
                    // auto e2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = e1 - s1;
                    std::cout << "search time: " << diff.count() / final_index_1->getQueryLen() << "\n";

                    float recall = 0;

                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        // if (res_1[i].size() == 0 or res_2[i].size() == 0)
                        if (res[i].size() == 0)
                            continue;
                        float tmp_recall = 0;
                        float cnt = 0;
                        for (unsigned j = 0; j < K; j++)
                        {
                            unsigned k = 0;
                            for (; k < K; k++)
                            {
                                // if (res_1[i][k] == final_index_2->getGroundData()[i * final_index_2->getGroundDim() + j])
                                //     break;
                                // if (res_2[i][k] == final_index_2->getGroundData()[i * final_index_2->getGroundDim() + j])
                                //     break;
                                if (res[i][k] == final_index_2->getGroundData()[i * final_index_2->getGroundDim() + j])
                                    break;
                            }
                            if (k == K)
                                cnt++;
                        }
                        tmp_recall = (float)(K - cnt) / (float)K;
                        recall = recall + tmp_recall;
                    }
                    // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                    float acc = recall / final_index_2->getQueryLen();
                    std::cout << K << " NN accuracy: " << acc << std::endl;
                }
            }
            e = std::chrono::high_resolution_clock::now();
            std::cout << "__SEARCH FINISH__" << std::endl;

            return this;
        }

        final_index_->getParam().set<unsigned>("K_search", K); // 这行代码设置了索引的参数K_search为K的值. 这意味着在后续的搜索中, 将寻找每个查询点的10个最近邻

        std::vector<std::vector<unsigned>> res;

        // ENTRY
        ComponentSearchEntry *a = nullptr;
        if (entry_type == SEARCH_ENTRY_NONE)
        {
            std::cout << "__SEARCH ENTRY : NONE__" << std::endl;
            a = new ComponentSearchEntryNone(final_index_);
        }
        else if (entry_type == SEARCH_ENTRY_CENTROID)
        {
            std::cout << "__SEARCH ENTRY : CENTROID__" << std::endl;
            a = new ComponentSearchEntryCentroid(final_index_);
        }
        else
        {
            std::cerr << "__SEARCH ENTRY : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        // ROUTE
        ComponentSearchRoute *b = nullptr;
        if (route_type == ROUTER_GREEDY)
        {
            std::cout << "__ROUTER : GREEDY__" << std::endl;
            b = new ComponentSearchRouteGreedy(final_index_);
        }
        else if (route_type == ROUTER_NSW)
        {
            std::cout << "__ROUTER : NSW__" << std::endl;
            b = new ComponentSearchRouteNSW(final_index_);
        }
        else if (route_type == ROUTER_HNSW)
        {
            std::cout << "__ROUTER : HNSW__" << std::endl;
            b = new ComponentSearchRouteHNSW(final_index_);
        }
        else if (route_type == ROUTER_DEG)
        {
            std::cout << "__ROUTER : DEG__" << std::endl;
            b = new ComponentSearchRouteDEG(final_index_);
            // b->UpdateEnterpointSet();
        }
        else
        {
            std::cerr << "__ROUTER : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        if (L_type == L_SEARCH_ASCEND)
        {
            std::set<unsigned> visited;
            unsigned sg = 1000;
            float acc_set = 0.9;
            bool flag = false;
            int L_sl = 1;
            unsigned L = 0;
            visited.insert(L);
            unsigned L_min = 0x7fffffff;
            for (unsigned t = 0; t < 20; t++)
            {

                L = L + K;
                std::cout << "SEARCH_L : " << L << std::endl;
                if (L < K)
                {
                    std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                    exit(-1);
                }

                final_index_->getParam().set<unsigned>("L_search", L);

                auto s1 = std::chrono::high_resolution_clock::now();

                res.clear();
                res.resize(final_index_->getQueryLen());
                //  #pragma omp parallel for
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    std::vector<Index::Neighbor> pool;
                    a->SearchEntryInner(i, pool);
                    b->RouteInner(i, pool, res[i]);
                }

                auto e1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e1 - s1;
                std::cout << "search time: " << diff.count() / final_index_->getQueryLen() << "\n";

                std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
                std::cout << "HopCount: " << final_index_->getHopCount() << std::endl;
                final_index_->resetDistCount();
                final_index_->resetHopCount();
                // int cnt = 0;
                float recall = 0;
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    if (res[i].size() == 0)
                        continue;
                    float tmp_recall = 0;
                    float cnt = 0;
                    for (unsigned j = 0; j < K; j++)
                    {
                        unsigned k = 0;
                        for (; k < K; k++)
                        {
                            if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                                break;
                        }
                        if (k == K)
                            cnt++;
                    }
                    tmp_recall = (float)(K - cnt) / (float)K;
                    recall = recall + tmp_recall;
                }
                float acc = recall / final_index_->getQueryLen();
                std::cout << K << " NN accuracy: " << acc << std::endl;
            }
        }
        e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }

    void IndexBuilder::peak_memory_footprint()
    {

        unsigned iPid = (unsigned)getpid();

        std::cout << "PID: " << iPid << std::endl;

        std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
        std::ifstream info(status_file);
        if (!info.is_open())
        {
            std::cout << "memory information open error!" << std::endl;
        }
        std::string tmp;
        while (getline(info, tmp))
        {
            if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
                std::cout << tmp << std::endl;
        }
        info.close();
    }
}