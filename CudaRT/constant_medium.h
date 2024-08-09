#pragma once

#include "renderable.h"
#include "texture.h"
#include "material.h"

class ConstantMedium {
public:
    ConstantMedium(Renderable* boundary, double density, Texture tex)
        : boundary(boundary)
        , neg_inv_density(-1 / density)
        , phase_function(Material::isotropic(tex))
    {}

    ConstantMedium(Renderable* boundary, float density, const Vec3f32& albedo)
        : boundary(boundary)
        , neg_inv_density(-1 / density)
        , phase_function(Material::isotropic(albedo))
    {}

    bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        HitRecord rec1, rec2;

        if (!boundary->hit(r, Interval::universe, rec1)) {
            return false;
        }

        if (!boundary->hit(r, Interval(rec1.t + 0.0001, INFINITY), rec2)) {
            return false;
        }

        if (rec1.t < ray_t.min) {
            rec1.t = ray_t.min;
        }
        if (rec2.t > ray_t.max) {
            rec2.t = ray_t.max;
        }

        if (rec1.t >= rec2.t) {
            return false;
        }

        if (rec1.t < 0) {
            rec1.t = 0;
        }

        auto ray_length = r.direction().length();
        auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        auto hit_distance = neg_inv_density * std::log(random_double());

        if (hit_distance > distance_inside_boundary) {
            return false;
        }

        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);

        rec.normal = Vec3f32(1, 0, 0);
        rec.front_face = true;
        rec.material = &phase_function;

        return true;
    }

private:
    Renderable* boundary;
    float neg_inv_density;
    // TODO: pointer
    Material phase_function;
};