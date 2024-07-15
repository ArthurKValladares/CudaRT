#pragma once

#include "vec3.h"
#include "rtw_image.h"
#include "interval.h"

enum class TextureType {
	SolidColor,
	CheckerPattern,
	Image
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

struct ImagePayload {
	RtwImage* image;
};

union TexturePayload {
	SolidColorPayload solid_color;
	CheckerPatternPayload checker_payload;
	ImagePayload image;

	__device__ TexturePayload& operator=(const TexturePayload& payload) {
		memcpy(this, &payload, sizeof(TexturePayload));

		return *this;
	}
};

struct TextureData {
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

	__device__ static TextureData Image(RtwImage* image) {
		TexturePayload payload = {};
		payload.image.image = image;
		return TextureData{
			TextureType::Image,
			payload
		};
	}

	__device__ TextureData& operator=(const TextureData& texture_data) {
		type = texture_data.type;
		payload = texture_data.payload;

		return *this;
	}

	TextureType type;
	TexturePayload payload;
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

	__device__ static Texture Image(RtwImage* image)
	{
		return Texture{
			TextureData::Image(image)
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
			case TextureType::Image: {
				const RtwImage& image = *data.payload.image.image;

				if (image.height() <= 0) {
					return Vec3f32(0.0, 1.0, 1.0);
				}

				u = Interval(0.0, 1.0).clamp(u);
				v = 1.0 - Interval(0.0, 1.0).clamp(v);

				auto i = int(u * image.width());
				auto j = int(v * image.height());
				auto pixel = image.pixel_data(i, j);

				auto color_scale = 1.0 / 255.0;
				return Vec3f32(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
			}
		}
	}

	TextureData data;
};