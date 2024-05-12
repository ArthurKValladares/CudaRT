#pragma once

#include "vec3.h"
#include "ray.h"

class hit_record {
public:
	vec3 p;
	vec3 normal;
	float t;
};