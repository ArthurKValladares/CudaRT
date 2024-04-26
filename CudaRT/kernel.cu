#include <iostream>
#include <time.h>
#include <fstream>

#include "vec3.h"
#include "ray.h"
#include "math.h"

#define SDL_MAIN_HANDLED
#include "SDL.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r) {
    const vec3 unit_direction = unit_vector(r.direction());
    const float t = 0.5f * (unit_direction.y() + 1.0f);
    return lerp(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(r);
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not be initialized!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Basic C SDL project", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        nx, ny, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window could not be created!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (
        fb, nx, ny,
        vec3(-2.0, -1.0, -1.0),
        vec3(4.0, 0.0, 0.0),
        vec3(0.0, 2.0, 0.0),
        vec3(0.0, 0.0, 0.0)
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as Image
    std::ofstream output_image;
    //txtOut.open ("Example.txt");
    output_image.open("output.ppm");
    output_image << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            output_image << ir << " " << ig << " " << ib << "\n";
        }
    }
    output_image.close();

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaFree(fb));

    return 0;
}