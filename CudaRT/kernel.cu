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
#include "texture.h"
#include "rtw_image.h"
#include "quad.h"

#define SDL_MAIN_HANDLED
#include "SDL.h"

#define MAX_BOUNCE_DEPTH 50
#define SAMPLES_PER_PIXEL 100

#define SPHERES_GRID_SIZE 5
#define SPHERE_COUNT (SPHERES_GRID_SIZE * 2 * SPHERES_GRID_SIZE * 2) + 1 + 3

// TODO: Should probably be in the camera class itself
#define CAMERA_DEFAULT_METERS_PER_SECOND 15.0
#define CAMERA_SPEED_DELTA 0.5

#define WORLD_IDX 5

__device__ Vec3f32 color(LocalRandomState& local_rand_state, HittableList* hittables, const Ray& r, Camera* cam, int max_bounces) {
    Ray curr_ray = r;
    Vec3f32 curr_color = Vec3f32(1, 1, 1);

    for (int i = 0; i < max_bounces; ++i) {
        HitRecord rec;

        if (hittables->hit(curr_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3f32 attenuation;
            const Vec3f32 color_from_emission = rec.material->emitted(rec.u, rec.v, rec.p);

            if (rec.material->scatter(r, rec, attenuation, scattered, local_rand_state)) {
                curr_color = color_from_emission + curr_color * attenuation;
                curr_ray = scattered;
            }
            else {
                return curr_color * color_from_emission;
            }
        }
        else {
            return curr_color * cam->background;
        }
    }

    return Vec3f32(0.0, 0.0, 0.0);
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

__global__ void random_state_init(int max_x, int max_y, curandState* rand_state) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    const unsigned long long seed = 1984;
    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(HittableList* hittables, curandState* rand_state, int ns, Uint32* fb, int max_x, int max_y, Camera* cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    LocalRandomState local_rand_state = LocalRandomState{ rand_state[pixel_index] };

    Vec3f32 col(0., 0., 0.);
    for (int s = 0; s < ns; s++) {
        float u = float(i + random_float(local_rand_state)) / float(max_x);
        float v = float(j + random_float(local_rand_state)) / float(max_y);
        Ray r = cam->get_ray(u, v, local_rand_state);
        col += color(local_rand_state, hittables, r, cam, MAX_BOUNCE_DEPTH);
    }

    fb[pixel_index] = vec3_to_color(col / float(ns));
}

__global__ void create_world_random_spheres(curandState* rand_state, Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        LocalRandomState local_rand_state = LocalRandomState{ rand_state[0] };

        int i = 0;

        renderables[i++] = Renderable::Sphere(Vec3f32(0, -1000.0, -1), 1000, Material::lambertian(
            Texture::CheckerPattern(0.32, Vec3f32(0.2, 0.3, 0.1), Vec3f32(0.9, 0.9, 0.9))
        ));

        for (int a = -SPHERES_GRID_SIZE; a < SPHERES_GRID_SIZE; a++) {
            for (int b = -SPHERES_GRID_SIZE; b < SPHERES_GRID_SIZE; b++) {
                const float choose_material = random_float(local_rand_state);
                const Vec3f32 center = Vec3f32(a + random_float(local_rand_state), 0.2, b + random_float(local_rand_state));

                if (choose_material < 0.8f) {
                    const float r = random_float(local_rand_state) * random_float(local_rand_state);
                    const float g = random_float(local_rand_state) * random_float(local_rand_state);
                    const float b = random_float(local_rand_state) * random_float(local_rand_state);

                    const Vec3f32 center2 = center + Vec3f32(0.0, random_float(local_rand_state, 0.0, 0.5), 0.0);
                    renderables[i++] = Renderable::Sphere(center, center2, 0.2,
                        Material::lambertian(Texture::SolidColor(Vec3f32(r, g, b))));
                }
                else if (choose_material < 0.95f) {
                    const float r = 0.5f * (1.0f + random_float(local_rand_state));
                    const float g = 0.5f * (1.0f + random_float(local_rand_state));
                    const float b = 0.5f * (1.0f + random_float(local_rand_state));
                    const float fuzz = 0.5f * random_float(local_rand_state);
                    
                    renderables[i++] = Renderable::Sphere(center, 0.2,
                        Material::metal(Vec3f32(r, g, b), fuzz));
                }
                else {
                    renderables[i++] = Renderable::Sphere(center, 0.2, Material::dieletric(1.5));
                }
            }
        }
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 1, 0), 1, Material::dieletric(1.5));
        renderables[i++] = Renderable::Sphere(Vec3f32(-4, 1, 0), 1, Material::lambertian(Texture::SolidColor(Vec3f32(0.4, 0.2, 0.1))));
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

__global__ void create_world_earth(Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny, RtwImage* image) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        int i = 0;
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 0, 0), 2, Material::lambertian(
            Texture::Image(image)
        ));

        *hittables = HittableList(renderables, i);

        Vec3f32 origin(0, 0, 12);
        Vec3f32 look_at(0, 0, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.1;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_world_perlin(curandState* rand_state, Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny, Perlin* perlin) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        LocalRandomState local_rand_state = LocalRandomState{ rand_state[0] };

        // TODO: This could be done better with more than one single block, in a separate step
        perlin->init(local_rand_state);

        int i = 0;
        renderables[i++] = Renderable::Sphere(Vec3f32(0, -1000, 0), 1000, Material::lambertian(
            Texture::Perlin(perlin, 4.0)
        ));
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 2, 0), 2, Material::lambertian(
            Texture::Perlin(perlin, 4.0)
        ));

        *hittables = HittableList(renderables, i);

        Vec3f32 origin(13, 2, 3);
        Vec3f32 look_at(0, 0, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.0;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_world_quads(Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        int i = 0;
        renderables[i++] = Renderable::Quad(Vec3f32(-3, -2, 5), Vec3f32(0, 0, -4), Vec3f32(0, 4,  0), Material::lambertian(1.0, 0.2, 0.2));
        renderables[i++] = Renderable::Quad(Vec3f32(-2, -2, 0), Vec3f32(4, 0,  0), Vec3f32(0, 4,  0), Material::lambertian(0.2, 1.0, 0.2));
        renderables[i++] = Renderable::Quad(Vec3f32( 3, -2, 1), Vec3f32(0, 0,  4), Vec3f32(0, 4,  0), Material::lambertian(0.2, 0.2, 1.0));
        renderables[i++] = Renderable::Quad(Vec3f32(-2,  3, 1), Vec3f32(4, 0,  0), Vec3f32(0, 0,  4), Material::lambertian(1.0, 0.5, 0.0));
        renderables[i++] = Renderable::Quad(Vec3f32(-2, -3, 5), Vec3f32(4, 0,  0), Vec3f32(0, 0, -4), Material::lambertian(0.2, 0.8, 0.8));

        *hittables = HittableList(renderables, i);

        Vec3f32 origin(0, 0, 9);
        Vec3f32 look_at(0, 0, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.0;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            80.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_world_simple_light(curandState* rand_state, Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny, Perlin* perlin) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        LocalRandomState local_rand_state = LocalRandomState{ rand_state[0] };

        // TODO: This could be done better with more than one single block, in a separate step
        perlin->init(local_rand_state);

        int i = 0;
        renderables[i++] = Renderable::Sphere(Vec3f32(0, -1000, 0), 1000, Material::lambertian(
            Texture::Perlin(perlin, 4.0)
        ));
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 7, 0), 2,
            Material::diffuse_light(Vec3f32(4.0, 4.0, 4.0))
        );
        renderables[i++] = Renderable::Sphere(Vec3f32(0, 2, 0), 2, Material::lambertian(
            Texture::Perlin(perlin, 4.0)
        ));

        renderables[i++] = Renderable::Quad(Vec3f32(3, 1, -2), Vec3f32(2, 0, 0), Vec3f32(0, 2, 0), 
            Material::diffuse_light(Vec3f32(4.0, 4.0, 4.0)));

        *hittables = HittableList(renderables, i);

        Vec3f32 origin(26, 3, 6);
        Vec3f32 look_at(0, 2, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.0;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            Vec3f32(0, 0, 0)
        );
    }
}

__device__ void create_box(const Vec3f32& a, const Vec3f32& b, Renderable* renderables, int& initial_idx, Material mat)
{
    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = Vec3f32(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
    auto max = Vec3f32(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

    auto dx = Vec3f32(max.x() - min.x(), 0, 0);
    auto dy = Vec3f32(0, max.y() - min.y(), 0);
    auto dz = Vec3f32(0, 0, max.z() - min.z());

    renderables[initial_idx++] = Renderable::Quad(Vec3f32(min.x(), min.y(), max.z()), dx, dy, mat);
    renderables[initial_idx++] = Renderable::Quad(Vec3f32(max.x(), min.y(), max.z()), -dz, dy, mat);
    renderables[initial_idx++] = Renderable::Quad(Vec3f32(max.x(), min.y(), min.z()), -dx, dy, mat);
    renderables[initial_idx++] = Renderable::Quad(Vec3f32(min.x(), min.y(), min.z()), dz, dy, mat);
    renderables[initial_idx++] = Renderable::Quad(Vec3f32(min.x(), max.y(), max.z()), dx, -dz, mat);
    renderables[initial_idx++] = Renderable::Quad(Vec3f32(min.x(), min.y(), min.z()), dx, dz, mat);
}

__global__ void create_world_cornell_box(Renderable* renderables, HittableList* hittables, Camera* d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        Material red   = Material::lambertian(0.65, 0.05, 0.05);
        Material white = Material::lambertian(0.73, 0.73, 0.73);
        Material green = Material::lambertian(0.12, 0.45, 0.15);
        Material light = Material::diffuse_light(Vec3f32(15, 15, 15));

        int i = 0;
        renderables[i++] = Renderable::Quad(Vec3f32(555, 0, 0), Vec3f32(0, 555, 0), Vec3f32(0, 0, 555), 
            green
        );
        renderables[i++] = Renderable::Quad(Vec3f32(0, 0, 0), Vec3f32(0, 555, 0), Vec3f32(0, 0, 555),
            red
        );
        renderables[i++] = Renderable::Quad(Vec3f32(343, 554, 332), Vec3f32(-130, 0, 0), Vec3f32(0, 0, -105),
            light
        );
        renderables[i++] = Renderable::Quad(Vec3f32(0, 0, 0), Vec3f32(555, 0, 0), Vec3f32(0, 0, 555),
            white
        );
        renderables[i++] = Renderable::Quad(Vec3f32(555, 555, 555), Vec3f32(-555, 0, 0), Vec3f32(0, 0, -555),
            white
        );
        renderables[i++] = Renderable::Quad(Vec3f32(0, 0, 555), Vec3f32(555, 0, 0), Vec3f32(0, 555, 0),
            white
        );


        *hittables = HittableList(renderables, i);

        Vec3f32 origin(278, 278, -1600);
        Vec3f32 look_at(278, 278, 0);
        float dist_to_focus = (origin - look_at).length();
        float aperture = 0.0;
        *d_camera = Camera(
            origin,
            look_at,
            Vec3f32(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            Vec3f32(0, 0, 0)
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

    RtwImage h_image = RtwImage("earthmap.jpg");
    unsigned char* d_bdata;
    RtwImage* d_image;
    const int total_bytes = h_image.get_total_bytes();
    {
        // Allocate device struct
        checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(RtwImage)));

        // Alocate device image data
        checkCudaErrors(cudaMalloc((void**)&d_bdata, total_bytes * sizeof(unsigned char)));

        // Copy host image data to device
        cudaMemcpy(d_bdata, h_image.bdata, sizeof(unsigned char) * total_bytes, cudaMemcpyHostToDevice);

        // Point to device data in host
        h_image.bdata = d_bdata;

        // Copy host struct to device
        cudaMemcpy(d_image, &h_image, sizeof(RtwImage), cudaMemcpyHostToDevice);
    }

    Perlin h_perlin = {};
    Vec3f32* randvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;
    Perlin* d_perlin;
    {
        // Allocate device struct
        checkCudaErrors(cudaMalloc((void**)&d_perlin, sizeof(Perlin)));

        // Alocate device Perlin data
        checkCudaErrors(cudaMalloc((void**)&randvec, Perlin::point_count * sizeof(Vec3f32)));
        checkCudaErrors(cudaMalloc((void**)&perm_x, Perlin::point_count * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&perm_y, Perlin::point_count * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&perm_z, Perlin::point_count * sizeof(int)));

        // Point to device data in host
        h_perlin.randvec = randvec;
        h_perlin.perm_x = perm_x;
        h_perlin.perm_y = perm_y;
        h_perlin.perm_z = perm_z;

        // Copy host struct to device
        cudaMemcpy(d_perlin, &h_perlin, sizeof(Perlin), cudaMemcpyHostToDevice);
    }

    Uint32* d_surface_buffer;
    checkCudaErrors(cudaMalloc(&d_surface_buffer, surface_buffer_size));

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    random_state_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Renderable* d_renderables;
    checkCudaErrors(cudaMalloc(&d_renderables, (SPHERE_COUNT + 1) * sizeof(Renderable)));
    HittableList* d_hittables;
    checkCudaErrors(cudaMalloc((void**)&d_hittables, sizeof(HittableList)));

    Camera* d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    Camera* h_camera = (Camera*) malloc(sizeof(Camera));

    switch (WORLD_IDX) {
    case 0: {
        create_world_random_spheres << <1, 1 >> > (d_rand_state, d_renderables, d_hittables, d_camera, nx, ny);
        break;
    }
    case 1: {
        create_world_earth << <1, 1 >> > (d_renderables, d_hittables, d_camera, nx, ny, d_image);
        break;
    }
    case 2: {
        create_world_perlin << <1, 1 >> > (d_rand_state, d_renderables, d_hittables, d_camera, nx, ny, d_perlin);
        break;
    }
    case 3: {
        create_world_quads << <1, 1 >> > (d_renderables, d_hittables, d_camera, nx, ny);
        break;
    }
    case 4: {
        create_world_simple_light << <1, 1 >> > (d_rand_state, d_renderables, d_hittables, d_camera, nx, ny, d_perlin);
        break;
    }
    case 5: {
        create_world_cornell_box << <1, 1 >> > (d_renderables, d_hittables, d_camera, nx, ny);
        break;
    }
    default: {
        printf("Invalid world id: %d!\n", WORLD_IDX);
        return 0;
    }
    }
    
    

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
                            update_camera << <1, 1 >> > (d_camera, -displacement * h_camera->front_movement_vector());
                            break;
                        }
                        case SDLK_a: {
                            update_camera << <1, 1 >> > (d_camera, -displacement * h_camera->right_movement_vector());
                            break;
                        }
                        case SDLK_s: {
                            update_camera << <1, 1 >> > (d_camera, displacement * h_camera->front_movement_vector());
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