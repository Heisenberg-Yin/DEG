#include <builder.h>
#include <set_para.h>
#include <iostream>


void HNSW(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0])
            ->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline1(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        parameters.set<float>("alpha", 0.5);
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        builder->peak_memory_footprint();

        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline2(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), true);
    builder->set_begin_time();
    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        auto *builder_1 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
        std::string graph_file_1 = graph_file + "_1";
        parameters.set<float>("alpha", 0);
        builder_1->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_1[0]);

        auto *builder_2 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));

        std::string graph_file_2 = graph_file + "_2";
        parameters.set<float>("alpha", 1);
        builder_2->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_2[0]);
        builder->set_end_time();
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        std::string graph_file_1 = graph_file + "_1";
        std::string graph_file_2 = graph_file + "_2";
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters, true);
        builder->peak_memory_footprint();

        builder->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file_1[0], &graph_file_2[0]);
        builder->peak_memory_footprint();

        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::DUAL_ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline3(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), true);
    builder->set_begin_time();

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        auto *builder_1 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
        std::string graph_file_1 = graph_file + "_1";
        parameters.set<float>("alpha", 0);
        builder_1->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_RTREE)
            ->save_graph(stkq::TYPE::INDEX_RTREE, &graph_file_1[0]);

        auto *builder_2 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));

        std::string graph_file_2 = graph_file + "_2";
        parameters.set<float>("alpha", 1);
        builder_2->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_2[0]);
        builder->set_end_time();           
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }

    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        std::string graph_file_1 = graph_file + "_1";
        std::string graph_file_2 = graph_file + "_2";
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters, true);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_RTREE_HNSW, &graph_file_1[0], &graph_file_2[0]);
        builder->peak_memory_footprint();
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_RTREE_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline4(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), true);

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_BS4)
            ->save_graph(stkq::TYPE::INDEX_BS4, &graph_file[0]);    
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }

    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->load_graph(stkq::TYPE::INDEX_BS4, &graph_file[0])
            ->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_BS4, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void DEG(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_DEG)
            ->save_graph(stkq::TYPE::INDEX_DEG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_DEG, &graph_file[0]);
        builder->peak_memory_footprint();
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

int main(int argc, char **argv)
{
    // ./test/main baseline1 openimage 0.5 1 1 build
    // ./test/main baseline2 openimage 0.5 1 1 build
    // ./test/main deg openimage 0.5 1 1 build

    if (argc != 7)
    {
        std::cout << "./main algorithm dataset alpha maximum_spatial_distance maximum_emb_distance exc_type"
                  << std::endl;
        exit(-1);
    }

    stkq::Parameters parameters;
    std::string dataset_root = R"(/mnt/hdd/yinziqi/yinziqi/graphann-tkq/dataset/)";
    std::string index_path = R"(/mnt/hdd/yinziqi/yinziqi/graphann-tkq/saved_index/)";
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<std::string>("index_path", index_path);
    parameters.set<unsigned>("n_threads", 8);

    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string alpha(argv[3]);
    std::string maximum_spatial_distance(argv[4]);
    std::string maximum_emb_distance(argv[5]);
    std::string exc_type(argv[6]);

    parameters.set<float>("alpha", std::stof(alpha));
    parameters.set<float>("max_spatial_distance", std::stof(maximum_spatial_distance));
    parameters.set<float>("max_emb_distance", std::stof(maximum_emb_distance));

    std::cout << "algorithm: " << alg << std::endl;
    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "max_emb_distance: " << maximum_emb_distance << std::endl;
    std::cout << "max_spatial_distance: " << maximum_spatial_distance << std::endl;
    std::string graph_file(alg + "_" + dataset + ".index");
    parameters.set<std::string>("graph_file", index_path + graph_file);
    parameters.set<std::string>("exc_type", exc_type);
    set_para(alg, dataset, parameters);

    if (alg == "baseline1")
    {
        baseline1(parameters);
    }
    else if (alg == "baseline2")
    {
        baseline2(parameters);
    }
    else if (alg == "baseline3")
    {
        baseline3(parameters);
    }
    else if (alg == "baseline4")
    {
        baseline4(parameters);
    }
    else if (alg == "deg")
    {
        DEG(parameters);
    }
    else
    {
        std::cout << "alg input error!\n";
        exit(-1);
    }
    return 0;
}