﻿#include <iostream>
#include <time.h>
#include <fstream>

#include "vec3.h"
#include "ray.h"
#include "math.h"
#include "sphere.h"

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

__device__ Uint32 color(Sphere** spheres, int sphere_count, const ray& r) {
    // TODO: For now hard-coded for 1 sphere
    const Sphere* sphere = *(spheres);

    vec3 ret_color;

    hit_record hr;
    if (sphere->hit(r, 0.0, 1.0, hr)) {
        vec3 N = unit_vector(r.at(hr.t) - vec3(0, 0, -1));
        ret_color = (0.5 * vec3(N.x() + 1, N.y() + 1, N.z() + 1)) * 255.99;
    }
    else {
        const vec3 unit_direction = unit_vector(r.direction());
        const float t = 0.5f * (unit_direction.y() + 1.0f);

        ret_color = lerp(vec3(0.5, 0.7, 1.0), vec3(1.0, 1.0, 1.0), t) * 255.99;
    }

    const Uint8 R = ret_color.r();
    const Uint8 G = ret_color.g();
    const Uint8 B = ret_color.b();

    return (0xFF << 24) | (R << 16) | (G << 8) | (B);
}

__global__ void render(Sphere** spheres, int sphere_count, Uint32* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(spheres, sphere_count, r);
}

__global__ void create_world(Sphere** spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(spheres) = new Sphere(vec3(0, 0, -1), 0.5);
    }
}

__global__ void free_world(Sphere** spheres) {
    delete* (spheres);
}

int main() {
    clock_t start, stop;

    int nx = 2400;
    int ny = 1200;
    int tx = 8;
    int ty = 8;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    const size_t surface_buffer_size = nx * ny * sizeof(Uint32);

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

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
    SDL_Surface* surface = SDL_CreateRGBSurface(0, nx, ny, 32,
        0x00FF0000,
        0x0000FF00,
        0x000000FF,
        0xFF000000
    );
    if (!surface) {
        printf("Surface could not be created!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }
    SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE);

    Uint32* surface_buffer;
    checkCudaErrors(cudaMalloc(&surface_buffer, surface_buffer_size));

    // TODO: Better create/free_world functions
    const int sphere_count = 1;
    Sphere** spheres;
    checkCudaErrors(cudaMalloc(&spheres, sphere_count * sizeof(Sphere*)));
    create_world << <1, 1 >> > (spheres);

    bool quit = false;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_KEYDOWN: {
                    switch (e.key.keysym.sym) {
                        case SDLK_ESCAPE: {
                            quit = true;
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                    break;
                }
                case SDL_QUIT: {
                    quit = true;
                    break;
                }
                default: {
                    break;
                }
            }
        }

        SDL_LockSurface(surface);
        {
            start = clock();

            render << <blocks, threads >> > (
                spheres, 1,
                surface_buffer, nx, ny,
                vec3(-2.0, -1.0, -1.0),
                vec3(4.0, 0.0, 0.0),
                vec3(0.0, 2.0, 0.0),
                vec3(0.0, 0.0, 0.0)
                );
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(surface->pixels, surface_buffer, surface_buffer_size, cudaMemcpyDeviceToHost));

            stop = clock();
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
            std::cerr << "took " << timer_seconds << " seconds.\n";
        }
        SDL_UnlockSurface(surface);

        if (SDL_BlitScaled(surface, nullptr, SDL_GetWindowSurface(window), nullptr))
        {
            printf("SDL_BlitScaled %s", SDL_GetError());
            exit(1);
        }

        SDL_UpdateWindowSurface(window);
    }

    // Cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (spheres);
    checkCudaErrors(cudaFree(surface_buffer));
    checkCudaErrors(cudaFree(spheres));

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}