#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// Function that catches errors

// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line)  {

        if (error != cudaSuccess) {
           printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
        }
}

// Has to be defined in the compilation in order to get the correct value of the
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Global variables

#define d 10
#define N 10
#define P 55

#define NUMBER_BLOCKS 512
#define THREADS_PER_BLOCKS 4*d

// Function that performs the product of Jacobi rotations
/* This function will be performed N times by N different blocks */

__global__ void Compute_all_rotations(float *J, float *A, float *out, const int *pos_i, const int *pos_j){

    __shared__ float temp[4 * d] ;  // variable that is to be shared by the threads in the block
    int block_j = blockDim.x * 4 * P ;
    int idx_J;
    int idx_A;
    int idx_out;

    for (int p=0 ; p<P ; p++) {
        // index = threadIdx.x + blockIdx.x*blockDim.x ;

        // Perform multiplications
        if (threadIdx.x % 4 == 0 ) {
            idx_J = 0 + 4 * p + block_j;
            idx_A = blockIdx.x * d * d + d * pos_i[p] + floorf(threadIdx.x / 4);
        }

        else if (threadIdx.x % 4 == 1 ) {
            idx_J = 1 + 4 * p + block_j;
            idx_A = blockIdx.x * d * d + d * pos_j[p] + floorf(threadIdx.x / 4);
        }

        else if (threadIdx.x % 4 == 2 ) {
            idx_J = 2 + 4 * p + block_j;
            idx_A = blockIdx.x * d * d + d * pos_i[p] + floorf(threadIdx.x / 4);
        }

        else if (threadIdx.x % 4 == 3 ) {
            idx_J = 3 + 4 * p + block_j;
            idx_A = blockIdx.x * d * d + d * pos_j[p] + floorf(threadIdx.x / 4);
        }

        temp[threadIdx.x] = J[idx_J] * A[idx_A] ;

        __syncthreads(); // synchronize threads

        // Perform additions

        if (threadIdx.x % 2 == 0){
            if (threadIdx.x % 4 == 0){
                idx_out = blockIdx.x * d * d + d * pos_i[p] + floorf(threadIdx.x / 4);
            }

            else if (threadIdx.x % 4 == 2){
                idx_out = blockIdx.x * d * d + d * pos_j[p] + floorf(threadIdx.x / 4);
            }
            out[idx_out] = temp[threadIdx.x] + temp[threadIdx.x + 1] ;
        }

        __syncthreads();  // synchronize threads
    }

}


// The folowing function reads the data stored in csv files and stores it in arrays

void read_file(const char * fname, float *array, int pos){

    FILE *file;
    char tampon[sizeof(float)];
    int actuel = 0;
    char c;
    int count;

    file = fopen (fname, "r");
    while ((c = fgetc(file)) != EOF) {
        if (c == ';' || c == '\n') {
            array[pos + count] = atof(tampon);
            actuel = 0;
            memset(tampon, 0, sizeof tampon);

        } else {
            tampon[actuel++] = c;
        }
    }

    printf("TEST\n");
    fclose (file);
}

void get_data(float *J, float *A){

    char fname[100] = {0};

    for (int n = 0 ; n<N ; n++){
        for (int p = 0 ; p<P ; p++){
            snprintf (fname, 100, "files/%i/J_%i.txt", n, p);
            read_file (fname, J, P*n*4 + 4*p);
        }

        snprintf (fname, 100, "files/%i/A.txt", n);
        read_file (fname, A, n*d*d);
    }
}


void write_result(float* out){

    FILE *file;
    const char* str = "; ";
    char fname[100] = {0};

    for (int n=0 ; n<N ; n++) {
        snprintf (fname, 100, "files/%i/out.txt", n);
        file = fopen(fname, "w");

        for (int i=0 ; i<d ; i++) {
            for (int j=0 ; j<d ; j++) {
                if (j == d-1) {
                  str = "\n";
                }

                fprintf(file, "%f %s", out[n*d*d + i*d +j], str);
                str = "; ";
            }
        }
        fclose(file);
    }
}

void positions(int* pos_i, int* pos_j){
    int shift = 0;

    for(int i=0 ; i<P ; i++){
        pos_i[i] = floor((i + shift) / d);
        pos_j[i] = (i + shift) % d;

        if((i + shift) % d == d-1){
            shift++;
        }
    }
}

int main(){

    // Properties of our GPUs

    cudaDeviceProp prop ;
    int count ;
    cudaGetDeviceCount(&count) ;

    for(int i=0 ; i<count ; i++) {
        cudaGetDeviceProperties(&prop, i) ;
        printf("Taille totale de la mémoire globale %ld\n", prop.totalGlobalMem) ;
    }

    // Define J A and out


    float J [P*4*d*N];
    float A [d*d*N];
    float out [d*d*N];

    get_data(J, A);

    // device copies

    float d_J [P*4*d*N];
    float d_A [d*d*N];
    float d_out [d*d*N];

    int size = sizeof(float);

    testCUDA(cudaMalloc((void **)&d_J, size));
    testCUDA(cudaMalloc((void **)&d_A, size));
    testCUDA(cudaMalloc((void **)&d_out, size));

    testCUDA(cudaMemcpy(d_A, &A, size, cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_J, &J, size, cudaMemcpyHostToDevice));

    // Define pos_i et pos_j

    int pos_i [P];
    int pos_j [P];

    int *d_pos_i, *d_pos_j;

    size = sizeof(int);

    testCUDA(cudaMalloc((void **)&d_pos_i, size));
    testCUDA(cudaMalloc((void **)&d_pos_j, size));

    positions(pos_i, pos_j);

    testCUDA(cudaMemcpy(d_pos_i, &pos_j, size, cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_pos_j, &pos_i, size, cudaMemcpyHostToDevice));

    // Timer definition and start

    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    // Computing rotations

    Compute_all_rotations<<<1,1>>>(d_J, d_A, d_out, d_pos_i, d_pos_j);

    // Stopping timer

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));
    printf("Exectudtion time: %f ms\n", TimerV);

    // Copying and saving result

    testCUDA(cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    write_result(out);

    // Cleanup

    testCUDA(cudaFree(d_A));
    testCUDA(cudaFree(d_J));
    testCUDA(cudaFree(d_out));
    testCUDA(cudaFree(d_pos_i));
    testCUDA(cudaFree(d_pos_j));
    testCUDA(cudaFree(start));
    testCUDA(cudaFree(stop));

    return 0;
}