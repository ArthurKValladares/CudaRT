#pragma once

#include "vec3.h"

enum class TextureType {
	SolidColor,
	CheckerPattern
};

struct SolidColorPayload {
	Vec3f32 albedo;
};

struct CheckerPatternPayload {
	float inv_scale;
	// TODO: for now this is just a color, in the future will be a `TextureData*` probably
	Vec3f32 even;
	Vec3f32 odd;
};

union TexturePayload {
	SolidColorPayload solid_color;
	CheckerPatternPayload checker_payload;
};

struct TextureData {
	TextureType type;
	TexturePayload payload;

	__device__ static TextureData SolidColor(const Vec3f32& albedo) {
		TexturePayload payload = {};
		payload.solid_color.albedo = albedo;
		return TextureData{
			TextureType::SolidColor,
			payload
		};
	}

	__device__ static TextureData CheckerPattern(float scale, const Vec3f32& color1, const Vec3f32& color2) {
		TexturePayload payload = {};
		payload.checker_payload.inv_scale = 1.0 / scale;
		payload.checker_payload.even = color1;
		payload.checker_payload.odd = color2;
		return TextureData{
			TextureType::CheckerPattern,
			payload
		};
	}
};

struct Texture {
	__device__ static Texture SolidColor(const Vec3f32& albedo) 
	{
		return Texture {
			TextureData::SolidColor(albedo)
		};
	}

	__device__ static Texture SolidColor(float red, float green, float blue)
	{
		return Texture{
			TextureData::SolidColor(Vec3f32(red, green, blue))
		};
	}

	__device__ static Texture CheckerPattern(float scale, const Vec3f32& color1, const Vec3f32& color2)
	{
		return Texture{
			TextureData::CheckerPattern(scale, color1, color2)
		};
	}

	__device__ Vec3f32 value(double u, double v, const Vec3f32& p) const {
		switch (data.type) {
			case TextureType::SolidColor: {
				return data.payload.solid_color.albedo;
			}
			case TextureType::CheckerPattern: {
				const float inv_scale = data.payload.checker_payload.inv_scale;

				const int x_integer = int(std::floor(inv_scale * p.x()));
				const int y_integer = int(std::floor(inv_scale * p.y()));
				const int z_integer = int(std::floor(inv_scale * p.z()));

				const bool isEven = (x_integer + y_integer + z_integer) % 2 == 0;

				return isEven ? data.payload.checker_payload.even : data.payload.checker_payload.odd;
			}
		}
	}

	TextureData data;
};