#include "index.h"

namespace stkq
{
    void RTreeIndex::query(Point const q, int k, float *base_loc_data, std::vector<unsigned> &result)
    {
        double lowerbound = 0;
        NodePriorityQueue node_queue;
        TopKPriorityQueue topk_queue(k);
        MyTree::Node *root = rtree.GetRoot(); // 获取RTree的根节点开始遍历
        // open the root node
        for (int index = 0; index < root->m_count; ++index)
        {
            node_queue.add_to_queue(root->m_branch[index].m_child, scoreMBR(&root->m_branch[index], q));
            // 遍历根节点的所有分支，将每个分支（子节点）根据它们的最小边界矩形（MBR）与查询点q的距离（由scoreMBR函数计算）加入到node_queue中
        }
        while (!node_queue.isEmpty())
        {
            // 当node_queue非空时，从队列中取出得分最低（距离最近）的节点
            auto const [node, min_score] = node_queue.toppop();
            // dequeue the best scored element
            if ((topk_queue.size() == k) && (min_score > topk_queue.peek()))
            {
                // 如果topk_queue已满（即已找到k个文档），并且当前节点的最小得分高于topk_queue中的最高得分，则可以剪枝
            }
            else if (node->IsInternalNode()) // is internal node
            {
                // add all children to the queue
                // 如果该节点是内部节点 遍历它的所有子节点 重复上述过程
                for (int i = 0; i < node->m_count; ++i)
                {
                    node_queue.add_to_queue(node->m_branch[i].m_child, scoreMBR(&node->m_branch[i], q));
                }
            }
            else // is leaf
            {
                // 如果是叶子节点 则计算其中每个文档与查询点q的距离，并更新topk_queue
                lowerbound = min_score;
                // 每当从叶子节点添加文档到topk_queue时，更新lowerbound为当前节点的最小得分
                for (int i = 0; i < node->m_count; i++)
                {
                    DocId const docid(node->m_branch[i].m_data);
                    // double score = distance(q, corp.docvec.at(docid), a, corp.max_space_distance, corp.max_semantic_distance);
                    double score = 0;
                    score = score + (*(base_loc_data + docid * 2) - q.first) * (*(base_loc_data + docid * 2) - q.first);
                    score = score + (*(base_loc_data + docid * 2 + 1) - q.second) * (*(base_loc_data + docid * 2 + 1) - q.second);

                    topk_queue.add_if_better(docid, std::sqrt(score));
                }
            }
            // 如果topk_queue的最高得分小于或等于lowerbound，则可以提前终止搜索，因为不会再找到更近的文档
            // very useful below
            if ((topk_queue.size() > 0) && (lowerbound >= topk_queue.peek()))
            {
                break;
            }
        }
        while (!topk_queue.isEmpty())
        {
            result.push_back(topk_queue.toppop().first);
        }
    }

    double RTreeIndex::scoreMBR(MyTree::Branch *a_branch, Point const q)
    {
        Point const mindistpoint = minSpaceDistPoint(*a_branch, q);
        double diff = 0;
        diff += (q.first - mindistpoint.first) * (q.first - mindistpoint.first);
        diff += (q.second - mindistpoint.second) * (q.second - mindistpoint.second);
        return std::sqrt(diff);
    }

    Point RTreeIndex::minSpaceDistPoint(MyTree::Branch const &a_branch, Point const &q) const
    {
        double px = q.first;
        double py = q.second;
        double rx, ry;

        if (px < a_branch.m_rect.m_min[0])
        {
            rx = a_branch.m_rect.m_min[0];
        }
        else if (px > a_branch.m_rect.m_max[0])
        {
            rx = a_branch.m_rect.m_max[0];
        }
        else
        {
            rx = px;
        }

        if (py < a_branch.m_rect.m_min[1])
        {
            ry = a_branch.m_rect.m_min[1];
        }
        else if (py > a_branch.m_rect.m_max[1])
        {
            ry = a_branch.m_rect.m_max[1];
        }
        else
        {
            ry = py;
        }

        return std::make_pair(rx, ry);
    }
}

// void RTreeIndex::PrintData()
// {
//     MyTree::Node *first = rtree.GetRoot();
//     PrintNodeData(first);
// }

// void RTreeIndex::PrintNodeData(const MyTree::Node *a_node)
// {
//     if (!(a_node->IsLeaf()))
//     {
//         for (int i = 0; i < a_node->m_count; ++i)
//         {
//             PrintNodeData(a_node->m_branch[i].m_child);
//         }
//         if (a_node)
//         {
//             if (a_node->IsInternalNode())
//             {
//                 std::cout << " - Internal node" << std::endl;
//             }
//             else
//             {
//                 std::cout << "  - Leaf node" << std::endl;
//             }
//             std::cout << "   with count = " << a_node->m_count << std::endl;
//             std::cout << "   with level = " << a_node->m_level << std::endl;
//             for (int index = 0; index < a_node->m_count; ++index)
//             {
//                 std::cout << "   with rect min = (" << a_node->m_branch[index].m_rect.m_min[0] << ","
//                           << a_node->m_branch[index].m_rect.m_min[1] << ")" << std::endl;
//                 std::cout << "   with rect max = (" << a_node->m_branch[index].m_rect.m_max[0] << ","
//                           << a_node->m_branch[index].m_rect.m_max[1] << ")" << std::endl;
//                 std::cout << "   with id = " << a_node->m_branch[index].id << std::endl;
//             }
//         }
//     }
//     else
//     {
//         if (a_node)
//         {
//             if (a_node->IsInternalNode())
//             {
//                 std::cout << " - Internal node" << std::endl;
//             }
//             else
//             {
//                 std::cout << "  - Leaf node" << std::endl;
//             }
//             std::cout << "   with count = " << a_node->m_count << std::endl;
//             std::cout << "   with level = " << a_node->m_level << std::endl;
//             for (int index = 0; index < a_node->m_count; ++index)
//             {
//                 std::cout << "   with rect min = (" << a_node->m_branch[index].m_rect.m_min[0] << ","
//                           << a_node->m_branch[index].m_rect.m_min[1] << ")" << std::endl;
//                 std::cout << "   with rect max = (" << a_node->m_branch[index].m_rect.m_max[0] << ","
//                           << a_node->m_branch[index].m_rect.m_max[1] << ")" << std::endl;
//                 std::cout << "   with data = " << a_node->m_branch[index].m_data << std::endl;
//                 std::cout << "   with id = " << a_node->m_branch[index].id << std::endl;
//             }
//         }
//     }
// }

// void RTreeIndex::PrintNodeData(const MyTree::Node *a_node, Doc q, double a, Corpus const &corp)
// {
//     if (!(a_node->IsLeaf()))
//     {
//         for (int i = 0; i < a_node->m_count; ++i)
//         {
//             PrintNodeData(a_node->m_branch[i].m_child, q, a, corp);
//         }
//         if (a_node)
//         {
//             if (a_node->IsInternalNode())
//             {
//                 std::cout << " - Internal node" << std::endl;
//             }
//             else
//             {
//                 std::cout << "  - Leaf node" << std::endl;
//             }
//             std::cout << "   with count = " << a_node->m_count << std::endl;
//             std::cout << "   with level = " << a_node->m_level << std::endl;
//             for (int index = 0; index < a_node->m_count; ++index)
//             {
//                 std::cout << "   with rect min = (" << a_node->m_branch[index].m_rect.m_min[0] << ","
//                           << a_node->m_branch[index].m_rect.m_min[1] << ")" << std::endl;
//                 std::cout << "   with rect max = (" << a_node->m_branch[index].m_rect.m_max[0] << ","
//                           << a_node->m_branch[index].m_rect.m_max[1] << ")" << std::endl;
//                 std::cout << "   with id = " << a_node->m_branch[index].id << std::endl;
//             }
//         }
//     }
//     else
//     {
//         if (a_node)
//         {
//             if (a_node->IsInternalNode())
//             {
//                 std::cout << " - Internal node" << std::endl;
//             }
//             else
//             {
//                 std::cout << "  - Leaf node" << std::endl;
//             }
//             std::cout << "   with count = " << a_node->m_count << std::endl;
//             std::cout << "   with level = " << a_node->m_level << std::endl;
//             for (int index = 0; index < a_node->m_count; ++index)
//             {
//                 std::cout << "   with rect min = (" << a_node->m_branch[index].m_rect.m_min[0] << ","
//                           << a_node->m_branch[index].m_rect.m_min[1] << ")" << std::endl;
//                 std::cout << "   with rect max = (" << a_node->m_branch[index].m_rect.m_max[0] << ","
//                           << a_node->m_branch[index].m_rect.m_max[1] << ")" << std::endl;
//                 std::cout << "   with data = " << a_node->m_branch[index].m_data << std::endl;
//                 std::cout << "   with score = " << distance(q, corp.docvec.at(a_node->m_branch[index].m_data), a, corp.max_space_distance, corp.max_semantic_distance) << std::endl;
//                 //		    std::cout << "   with WESscore = " << distance_with_equal_semantics(mindistpoint, q, a, corp.max_space_distance);
//                 std::cout << "   with id = " << a_node->m_branch[index].id << std::endl;
//             }
//         }
//     }
// }