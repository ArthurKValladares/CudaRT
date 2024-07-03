#include <iostream>
#include <time.h>
#include <fstream>

#include <curand_kernel.h>

#include "defs.h"
#include "vec3.h"
#include "ray.h"
#include "math.h"
#include "renderable.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

#define SDL_MAIN_HANDLED
#include "SDL.h"

#define MAX_BOUNCE_DEPTH 5
#define SAMPLES_PER_PIXEL 15

#define SPHERES_GRID_SIZE 5
#define SPHERE_COUNT (SPHERES_GRID_SIZE * 2 * SPHERES_GRID_SIZE * 2) + 1 + 3

// TODO: Should probably be in the camera class itself
#define CAMERA_DEFAULT_METERS_PER_SECOND 15.0
#define CAMERA_SPEED_DELTA 0.5

__device__ Vec3f32 color(curandState* local_rand_state, HittableList* hittables, const Ray& r) {
    Ray cur_ray = r;
    Vec3f32 cur_attenuation = Vec3f32(1.0, 1.0, 1.0);
    for (int i = 0; i < MAX_BOUNCE_DEPTH; i++) {
        HitRecord rec;
        if (hittables->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3f32 attenuation;
            if (rec.material->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return Vec3f32(0.0, 0.0, 0.0);
            }
        }
        else {
            Vec3f32 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3f32 c = lerp(Vec3f32(1.0, 1.0, 1.0), Vec3f32(0.5, 0.7, 1.0), t);
            return cur_attenuation * c;
        }
    }
    return Vec3f32(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

__device__ Vec3f32 linear_to_gamma(Vec3f32 linear_vec)
{
   return Vec3f32(linear_to_gamma(linear_vec.x()), linear_to_gamma(linear_vec.y()), linear_to_gamma(linear_vec.z()));
}

__device__ Uint32 vec3_to_color(Vec3f32 color) {
    color = linear_to_gamma(color);

    color *= 255.99;

    const Uint8 R = color.r();
    const Uint8 G = color.g();
    const Uint8 B = color.b();

    return (0xFF << 24) | (R << 16) | (G << 8) | (B);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    const unsigned long long seed = 1984;
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(HittableList* hittables, curandState* rand_state, int ns, Uint32* fb, int max_x, int max_y, Camera* cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    Vec3f32 col(0., 0., 0.);
    for (int s = 0; s < ns; s++) {
        float u = float(i + random_float(&local_rand_state)) / float(max_x);
        float v = float(j + random_float(&local_rand_state)) / float(max_y);
        Ray r = cam->get_ray(u, v, rand_state);
        col += color(&local_rand_state, hittables, r);
    }

    fb[pixel_index] = vec3_to_color(col / float(ns));
}

__global__ void create_world(curandState* rand_state, Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        renderables[i++] = Renderable::Sphere(Vec3f32(0, -1000.0, -1), 1000, Material::lambertian(Vec3f32(0.5, 0.5, 0.)));
        for (int a = -SPHERES_GRID_SIZE; a < SPHERES_GRID_SIZE; a++) {
            for (int b = -SPHERES_GRID_SIZE; b < SPHERES_GRID_SIZE; b++) {
                const float choose_material = random_float(rand_state);
                const Vec3f32 center = Vec3f32(a + random_float(rand_state), 0.2, b + random_float(rand_state));

                if (choose_material < 0.8f) {
                    const float r = random_float(rand_state) * random_float(rand_state);
                    const float g = random_float(rand_state) * random_float(rand_state);
                    const float b = random_float(rand_state) * random_float(rand_state);

                    const Vec3f32 center2 = center + Vec3f32(0.0, random_float(rand_state, 0.0, 0.5), 0.0);
                    renderables[i++] = Renderable::Sphere(center, center2, 0.2,
                        Material::lambertian(Vec3f32(r, g, b)));
                }
                else if (choose_material < 0.95f) {
                    const float r = 0.5f * (1.0f + random_float(rand_state));
                    const float g = 0.5f * (1.0f + random_float(rand_state));
                    const float b = 0.5f * (1.0f + random_float(rand_state));
                    const float fuzz = 0.5f * random_float(rand_state);
                    
                    renderables[i++] = Renderable::Sphere(center, 0.2,
                        Material::metal(Vec3f32(r, g, b), fuzz));
                }
                else {
                    renderables[i++] = Renderable::Sphere(center, 0.2, Material::dieletric(1.5));
                }
            }
        }
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 1, 0), 1, Material::dieletric(1.5));
        renderables[i++] = Renderable::Sphere(Vec3f32(-4, 1, 0), 1, Material::lambertian(Vec3f32(0.4, 0.2, 0.1)));
        renderables[i++] = Renderable::Sphere(Vec3f32(4, 1, 0), 1, Material::metal(Vec3f32(0.7, 0.6, 0.5), 0.0));

        assert(SPHERE_COUNT == i);
        *hittables = HittableList(renderables, i);

        Vec3f32 origin(13, 3, 3);
        Vec3f32 look_at(0, 0, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.1;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void update_camera(Camera* d_camera, Vec3f32 displacement) {
    d_camera->update_position(displacement);
}

int main() {
    clock_t start, stop;

    int nx = 2400 / 3;
    int ny = 1200 / 3;

    int num_pixels = nx * ny;

    const size_t surface_buffer_size = nx * ny * sizeof(Uint32);

    // Block stuff
    int tx = 8;
    int ty = 8;
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

    Uint32* d_surface_buffer;
    checkCudaErrors(cudaMalloc(&d_surface_buffer, surface_buffer_size));

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    Renderable* d_renderables;
    checkCudaErrors(cudaMalloc(&d_renderables, (SPHERE_COUNT + 1) * sizeof(Renderable)));
    HittableList* d_hittables;
    checkCudaErrors(cudaMalloc((void**)&d_hittables, sizeof(HittableList)));
    Camera* d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    Camera* h_camera = (Camera*) malloc(sizeof(Camera));

    create_world << <1, 1 >> > (d_rand_state, d_renderables, d_hittables, d_camera, nx, ny);

    printf("Starting Rendering!\n");
    float camera_meters_per_second = CAMERA_DEFAULT_METERS_PER_SECOND;
    double timer_seconds = 0.0;
    bool quit = false;
    while (!quit) {
        checkCudaErrors(cudaMemcpy(h_camera, d_camera, sizeof(Camera), cudaMemcpyDeviceToHost));

        const float displacement = camera_meters_per_second * timer_seconds;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_KEYDOWN: {
                    switch (e.key.keysym.sym) {
                        case SDLK_ESCAPE: {
                            quit = true;
                            break;
                        }
                        case SDLK_w: {
                            update_camera << <1, 1 >> > (d_camera, displacement * h_camera->front_movement_vector());
                            break;
                        }
                        case SDLK_a: {
                            update_camera << <1, 1 >> > (d_camera, -displacement * h_camera->right_movement_vector());
                            break;
                        }
                        case SDLK_s: {
                            update_camera << <1, 1 >> > (d_camera, -displacement * h_camera->front_movement_vector());
                            break;
                        }
                        case SDLK_d: {
                            update_camera << <1, 1 >> > (d_camera, displacement * h_camera->right_movement_vector());
                            break;
                        }
                        case SDLK_SPACE: {
                            update_camera << <1, 1 >> > (d_camera, Vec3f32(0.0, displacement, 0.0));
                            break;
                        }
                        case SDLK_LSHIFT: {
                            update_camera << <1, 1 >> > (d_camera, Vec3f32(0.0, -displacement, 0.0));
                            break;
                        }
                        // '+'
                        case SDLK_EQUALS: {
                            camera_meters_per_second += CAMERA_SPEED_DELTA;
                            break;
                        }
                        case SDLK_MINUS: {
                            if (camera_meters_per_second - CAMERA_SPEED_DELTA > 0.) {
                                camera_meters_per_second -= CAMERA_SPEED_DELTA;
                            }
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
                d_hittables,
                d_rand_state, SAMPLES_PER_PIXEL,
                d_surface_buffer, nx, ny,
                d_camera
            );
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(surface->pixels, d_surface_buffer, surface_buffer_size, cudaMemcpyDeviceToHost));

            stop = clock();
            timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
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
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_surface_buffer));
    checkCudaErrors(cudaFree(d_hittables));
    checkCudaErrors(cudaFree(d_camera));

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}