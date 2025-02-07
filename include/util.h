#ifndef STKQ_UTIL_H
#define STKQ_UTIL_H

#include <random>
#include <algorithm>

namespace stkq {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        // 此函数的目的是在给定范围内生成一组不重复的随机数 并将这些数存储在一个数组中 下面是对这段代码的详细解释
        // rng 随机数生成器
        // addr 指向要存储生成的随机数的数组的指针
        // size 要生成的随机数的数量
        // N 随机数生成的范围上限
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        // 首先，函数在 [0, N - size) 范围内生成 size 个随机数。因为范围是 N - size，这样做是为了避免生成重复的数
        std::sort(addr, addr + size);
        //对这些生成的数进行排序
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        // 接下来，通过遍历数组并比较相邻元素来确保所有的数都是唯一的。如果发现相邻元素相同，则将当前元素设置为前一个元素加一，从而确保每个数字都是唯一的
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
        // 最后，函数计算一个 [0, N) 范围内的随机偏移量 off，并将这个偏移量应用到数组中的每个元素上 
        // 这是通过取 (addr[i] + off) % N 实现的，确保结果仍然在 [0, N) 范围内
    }
}
#endif