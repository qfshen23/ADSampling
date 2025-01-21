#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf/ivf.h>
#include <adsampling.h>
#include <getopt.h>

using namespace std;

const int MAXK = 100;

long double rotation_time=0;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k){
    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;

    vector<int> nprobes;
    // nprobes.push_back(10);
    // nprobes.push_back(20);
    // nprobes.push_back(50);
    // nprobes.push_back(80);
    // nprobes.push_back(100);
    nprobes.push_back(200);
    
    for(auto nprobe:nprobes){
        total_time=0;
        adsampling::clear();
        int correct = 0;

        for(int i=0;i<Q.n;i++){
            GetCurTime( &run_start);
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, k, nprobe);
            GetCurTime( &run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
            // Recall
            while(KNNs.empty() == false){
                int id = KNNs.top().second;
                KNNs.pop();
                for(int j=0;j<k;j++)
                    if(id == G.data[i * G.d + j]) correct ++;
            }
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);
        
        // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
        cout << "---------------ADSampling------------------------" << endl;
        cout << "nprobe = " << nprobe << " k = " << k <<  endl;
        cout << "Recall = " << recall * 100.000 << "%\t" << endl;
        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        // cout << "total_time: " << total_time << ", time1 = " << adsampling::time1 << " time2 = " << adsampling::time2 - adsampling::time3 << " time3 = " << adsampling::time3 << endl;
        cout << "time1 proportion: " << adsampling::time1 / total_time * 100 << "%, time2 proportion: " << (adsampling::time2) / total_time * 100 << "%, time3 proportion: " << adsampling::time3 / total_time * 100 << ", other" << (total_time - adsampling::time1 - adsampling::time2 - adsampling::time3) / total_time * 100 << "%" << endl;
        cout << "time1: " << adsampling::time1 << ", time2: " << adsampling::time2 << ", time3: " << adsampling::time3 << endl;
        cout << "average count of exact distance vectors: " << adsampling::cntt / Q.n << endl;
        cout << "average count of exact srq_dist calls: " << adsampling::dist_cnt / Q.n << endl;
        cout << "total dimention: " << adsampling::tot_dimension << endl;
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"K",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"delta_d",                     required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
        {"propotion",                   required_argument, 0, 'a'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    // getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";

    int randomize = 0;
    int subk = 100;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if(optarg) subk = atoi(optarg);
                break;  
            case 'e':
                if(optarg)adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)adsampling::delta_d = atoi(optarg);
                break;              
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
        }
    }
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    
    freopen(result_path, "a", stdout);

    if(randomize) {
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }
    
    IVF ivf;
    ivf.load(index_path);
    test(Q, G, ivf, subk);
    return 0;
}
