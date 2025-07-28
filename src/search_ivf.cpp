#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
// #define COUNT_DIMENSION
// #define PLOT_DISK_K
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

char diskK_path[256] = "";

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k, int k_overlap, int refine_num){
    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;

    vector<pair<int, int>> test_params;
    
    test_params.push_back({12, 4000});
    
    
#ifdef PLOT_DISK_K
    std::ofstream fout(diskK_path);
    if(!fout.is_open()) {
        std::cerr << "Error: cannot open file " << diskK_path << std::endl;
        exit(1);
    }
#endif
    
    for(auto params : test_params) {
        int nprobe = params.first;
        int curr_refine_num = params.second;
        total_time=0;
        adsampling::clear();
        int correct = 0;

        for(int i=0;i<Q.n;i++){
#ifdef PLOT_DISK_K
            adsampling::diskK_vec.clear();
#endif
            GetCurTime( &run_start);
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, k, nprobe, 0, k_overlap, curr_refine_num);
            GetCurTime( &run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

#ifdef PLOT_DISK_K
            // plot diskK curve
            if((i % 1000) == 0) {              
                fout << adsampling::diskK_vec.size() << " " << i << std::endl;
                for(auto x: adsampling::diskK_vec) {
                    fout << x << " ";
                }
                fout << std::endl;
            }
#endif

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
        cout << "nprobe = " << nprobe << " k = " << k << " refine_num = " << curr_refine_num << endl;
        cout << "Recall = " << recall * 100.000 << "%\t" << endl;
        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << "total distance calculation: " << adsampling::dist_cnt << endl;
        cout << "time1: " << adsampling::time1 << ", time2: " << adsampling::time2 << ", time3: " << adsampling::time3 << ", time4: " << adsampling::time4 << endl;
    }
#ifdef PLOT_DISK_K
    fout.close();
#endif
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
        {"diskK_path",                  required_argument, 0, 'a'},
        {"k_overlap",                   required_argument, 0, 'o'},
        {"refine_num",                  required_argument, 0, 'r'},
        {"topk_clusters_path",          required_argument, 0, 'b'},
        {"cc",                          required_argument, 0, 'f'},
        {"top_centroids_path",          required_argument, 0, 'h'},
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
    char topk_clusters_path[256] = "";
    int randomize = 0;
    int subk = 1;
    int k_overlap = 0;
    int refine_num = 0;
    int cc = 0;
    char top_centroids_path[256] = "";
    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:a:o:c:b:f:h:", longopts, &ind);
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
            case 'a':
                if(optarg)strcpy(diskK_path, optarg);
                break;
            case 'o':
                if(optarg)k_overlap = atoi(optarg);
                break;
            case 'c':
                if(optarg)refine_num = atoi(optarg);
                break;
            case 'b':
                if(optarg)strcpy(topk_clusters_path, optarg);
                break;
            case 'f':
                if(optarg)cc = atoi(optarg);
                break;
            case 'h':
                if(optarg)strcpy(top_centroids_path, optarg);
                break;
        }
    }
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    
    if(freopen(result_path, "a", stdout) == NULL) {
        std::cerr << "Error: cannot open file " << result_path << std::endl;
        exit(1);
    }

    if(randomize) {
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }
    
    IVF ivf;
    ivf.load(index_path);

    if (topk_clusters_path != "") {
        ivf.setTopkCentroidsNum(cc);
        ivf.loadTopkCentroids(top_centroids_path);
        ivf.loadTopkClusters(topk_clusters_path, k_overlap);
        ivf.flattenTopkClusters();
    }
    test(Q, G, ivf, subk, k_overlap, refine_num);
    return 0;
}
