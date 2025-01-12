#include "parameters.h"
#include <string.h>
#include <iostream>

void NSW_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned max_m0, ef_construction;
    if (dataset == "Twitter10K")
    {
        max_m0 = 40, ef_construction = 200; // sift1M
    }
    else if (dataset == "coco")
    {
        max_m0 = 40, ef_construction = 200;
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("NN", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
}

void HNSW_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned max_m, max_m0, ef_construction;
    if (dataset == "Twitter10K")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else if (dataset == "coco")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else if (dataset == "sg-ins")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else if (dataset == "openimage")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }else if (dataset == "cc3m"){
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }else if (dataset == "Twitter10M")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }else if (dataset == "vediotext1m"){
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }else if (dataset == "audiovedio1M"){
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }else if (dataset == "howto100m"){
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("max_m0", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
    parameters.set<int>("mult", -1);
}

void DEG_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned max_m, init_edge, ef_construction, candidate_edge, ITER, rnn_size, update_layer, L_refine, R_refine, C_refine;
    if (dataset == "Twitter10K")
    {
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }
    else if (dataset == "coco")
    {
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }
    else if (dataset == "sg-ins")
    {
        // max_m = 40, max_m0 = 40, ef_construction = 200;
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }
    else if (dataset == "openimage")
    {
        // max_m = 40, max_m0 = 40, ef_construction = 200;
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }else if (dataset == "cc3m"){
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }else if (dataset == "Twitter10M")
    {
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;
    }else if (dataset == "vediotext1m"){
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;        
    }else if (dataset == "audiovedio1M"){
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;        
    }else if (dataset == "howto100m"){
        max_m = 40, init_edge = 10, ITER = 6, candidate_edge = 50, update_layer = 1, rnn_size = 100, L_refine = 100, R_refine = 50, C_refine = 500, ef_construction = 200;        
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("ef_construction", ef_construction);
    parameters.set<unsigned>("init_edge", init_edge);
    parameters.set<unsigned>("ITER", ITER);
    parameters.set<unsigned>("candidate_edge", candidate_edge);
    parameters.set<unsigned>("update_layer", update_layer);
    parameters.set<unsigned>("rnn_size", rnn_size);
    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<unsigned>("C_refine", C_refine);
    parameters.set<int>("mult", -1);
}

void set_data_path(std::string dataset, stkq::Parameters &parameters)
{
    // dataset root path
    std::string dataset_root = parameters.get<std::string>("dataset_root");
    std::string base_emb_path(dataset_root);
    std::string base_loc_path(dataset_root);
    std::string query_emb_path(dataset_root);
    std::string query_loc_path(dataset_root);
    std::string partition_path(dataset_root);
    std::string ground_path(dataset_root);
    if (dataset == "Twitter10K")
    {
        /// mnt/newpart/yinziqi/graphann-tkq/data/Twitter10K/base_loc.fvecs
        base_emb_path.append(R"(Twitter10K/base_emb.fvecs)");
        base_loc_path.append(R"(Twitter10K/base_loc.fvecs)");
        query_emb_path.append(R"(Twitter10K/query_emb.fvecs)");
        query_loc_path.append(R"(Twitter10K/query_loc.fvecs)");
        // partition_path.append(R"(Twitter10K/partition_results.txt)");
        ground_path.append(R"(Twitter10K/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }
    else if (dataset == "coco")
    {
        base_emb_path.append(R"(coco/base_img_emb.fvecs)");
        base_loc_path.append(R"(coco/base_text_emb.fvecs)");
        query_emb_path.append(R"(coco/query_img_emb.fvecs)");
        query_loc_path.append(R"(coco/query_text_emb.fvecs)");
        ground_path.append(R"(coco/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }
    else if (dataset == "sg-ins")
    {
        base_emb_path.append(R"(SG-ins/base_emb.fvecs)");
        base_loc_path.append(R"(SG-ins/base_loc.fvecs)");
        query_emb_path.append(R"(SG-ins/query_emb.fvecs)");
        query_loc_path.append(R"(SG-ins/new_query_loc.fvecs)");
        ground_path.append(R"(SG-ins/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }
    else if (dataset == "openimage")
    {
        // base_emb_path.append(R"(OpenImage/sample_base_img_emb.fvecs)");
        // base_loc_path.append(R"(OpenImage/sample_base_text_emb.fvecs)");
        base_emb_path.append(R"(OpenImage/base_img_emb.fvecs)");
        base_loc_path.append(R"(OpenImage/base_text_emb.fvecs)");
        query_emb_path.append(R"(OpenImage/query_img_emb.fvecs)");
        query_loc_path.append(R"(OpenImage/query_text_emb.fvecs)");
        ground_path.append(R"(OpenImage/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }else if (dataset == "cc3m"){
        base_emb_path.append(R"(CC3M/base_img_emb.fvecs)");
        base_loc_path.append(R"(CC3M/base_text_emb.fvecs)");
        query_emb_path.append(R"(CC3M/query_img_emb.fvecs)");
        query_loc_path.append(R"(CC3M/query_text_emb.fvecs)");
        ground_path.append(R"(CC3M/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");        
    }else  if (dataset == "Twitter10M")
    {
        /// mnt/newpart/yinziqi/graphann-tkq/data/Twitter10K/base_loc.fvecs
        base_emb_path.append(R"(Twitter10M/base_emb.fvecs)");
        base_loc_path.append(R"(Twitter10M/sample_base_loc.fvecs)");
        query_emb_path.append(R"(Twitter10M/query_emb.fvecs)");
        query_loc_path.append(R"(Twitter10M/query_loc.fvecs)");
        // partition_path.append(R"(Twitter10K/partition_results.txt)");
        ground_path.append(R"(Twitter10M/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }else if (dataset == "vediotext1m")
    {
        /// mnt/newpart/yinziqi/graphann-tkq/data/Twitter10K/base_loc.fvecs
        base_emb_path.append(R"(videotext1M/base_img_emb.fvecs)");
        base_loc_path.append(R"(videotext1M/base_text_emb.fvecs)");
        query_emb_path.append(R"(videotext1M/query_img_emb.fvecs)");
        query_loc_path.append(R"(videotext1M/query_text_emb.fvecs)");
        // partition_path.append(R"(Twitter10K/partition_results.txt)");
        ground_path.append(R"(videotext1M/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }else if (dataset == "audiovedio1M")
    {
        /// mnt/newpart/yinziqi/graphann-tkq/data/Twitter10K/base_loc.fvecs
        base_emb_path.append(R"(audiovedio1M/base_img_emb.fvecs)");
        base_loc_path.append(R"(audiovedio1M/base_text_emb.fvecs)");
        query_emb_path.append(R"(audiovedio1M/query_img_emb.fvecs)");
        query_loc_path.append(R"(audiovedio1M/query_text_emb.fvecs)");
        // partition_path.append(R"(Twitter10K/partition_results.txt)");
        ground_path.append(R"(audiovedio1M/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }else if (dataset == "howto100m"){
        /// mnt/newpart/yinziqi/graphann-tkq/data/Twitter10K/base_loc.fvecs
        base_emb_path.append(R"(howto100m/base_img_emb.fvecs)");
        base_loc_path.append(R"(howto100m/base_text_emb.fvecs)");
        query_emb_path.append(R"(howto100m/query_img_emb.fvecs)");
        query_loc_path.append(R"(howto100m/query_text_emb.fvecs)");
        // partition_path.append(R"(Twitter10K/partition_results.txt)");
        ground_path.append(R"(howto100m/)" + parameters.get<std::string>("alpha") + "_top10_results.ivecs");
    }
    else
    {
        std::cout << "dataset input error!\n";
        exit(-1);
    }
    parameters.set<std::string>("base_emb_path", base_emb_path);
    parameters.set<std::string>("base_loc_path", base_loc_path);
    parameters.set<std::string>("query_emb_path", query_emb_path);
    parameters.set<std::string>("query_loc_path", query_loc_path);
    parameters.set<std::string>("partition_path", partition_path);
    parameters.set<std::string>("ground_path", ground_path);
}

void NSG_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned K, L, Iter, S, R, L_refine, R_refine, C;
    if (dataset == "Twitter10K")
    {
        K = 25, L = 50, Iter = 10, S = 10, R = 100, L_refine = 100, R_refine = 40, C = 500; // nsg
    }
    else
    {
        std::cout << "input dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);
    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<unsigned>("C_refine", C);
}

void set_para(std::string alg, std::string dataset, stkq::Parameters &parameters)
{

    set_data_path(dataset, parameters);
    if (parameters.get<std::string>("exc_type") != "build")
    {
        return;
    }
    if (alg == "nsw")
    {
        NSW_PARA(dataset, parameters);
    }
    else if (alg == "nswv2")
    {
        NSWV2_PARA(dataset, parameters);
    }
    else if (alg == "hnsw")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (alg == "nsg")
    {
        NSG_PARA(dataset, parameters);
    }
    else if (alg == "DEG")
    {
        DEG_PARA(dataset, parameters);
    }  
    else if (alg == "baseline1")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (alg == "baseline2")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (alg == "baseline3")
    {
        HNSW_PARA(dataset, parameters);
    }    
    else
    {
        std::cout << "algorithm input error!\n";
        exit(-1);
    }
}
