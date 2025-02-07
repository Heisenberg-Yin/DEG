#ifndef STKQ_DISTANCE_H
#define STKQ_DISTANCE_H

namespace stkq {
    class E_Distance {
    public:
        template<typename T>
        T compare(const T *a, const T *b, unsigned length) const {            
            T emb_distance = 0;
            if (length < 4){
                for (int i = 0; i < length; i++){
                    emb_distance = emb_distance + (a[i] - b[i]) * (a[i] - b[i]);
                }
                return std::sqrt(emb_distance) / max_emb_dist;
            }

            float diff0, diff1, diff2, diff3;
            const T *last = a + length;
            const T *unroll_group = last - 3;

            /* Process 4 items with each loop for efficiency. */
            while (a < unroll_group) {
                diff0 = a[0] - b[0];
                diff1 = a[1] - b[1];
                diff2 = a[2] - b[2];
                diff3 = a[3] - b[3];
                emb_distance += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                a += 4;
                b += 4;
            }
            /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
            while (a < last) {
                diff0 = *a++ - *b++;
                emb_distance += diff0 * diff0;
            }
            //计算的是两个向量之间的欧几里得距离的平方，但没有进行开根号操作
            return std::sqrt(emb_distance) / max_emb_dist;
        }
        E_Distance(float max_emb_dist): max_emb_dist(max_emb_dist) {}
    private:
        float max_emb_dist = 0;
    };

    class S_Distance {
    public:
        template<typename T>
        T compare(const T *a, const T *b, unsigned length) const {
            T spatial_distance = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
            return std::sqrt(spatial_distance)/max_spatial_dist;
            // return (spatial_distance/max_spatial_dist);
        }
        S_Distance(float max_spatial_dist): max_spatial_dist(max_spatial_dist) {}
    private:
        float max_spatial_dist = 0;
    };
} 

#endif 
