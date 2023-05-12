/*
 * Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
 * holder of all proprietary rights on this computer program.
 * You can only use this computer program if you have closed
 * a license agreement with MPG or you get the right to use the computer
 * program from someone who is authorized to grant you that right.
 * Any use of the computer program without a valid license is prohibited and
 * liable to prosecution.
 *
 * Copyright©2019 Max-Planck-Gesellschaft zur Förderung
 * der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
 * for Intelligent Systems. All rights reserved.
 *
 * @author Vasileios Choutas
 * Contact: vassilis.choutas@tuebingen.mpg.de
 * Contact: ps-license@tuebingen.mpg.de
 *
 */

#ifndef AABB_H
#define AABB_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "defs.hpp"
#include "device_launch_parameters.h"
#include "double_vec_ops.h"
#include "helper_math.h"
#include "math_utils.hpp"

#define EPSILON 0.000001

template <typename T>
__align__(32) struct AABB {
   public:
    __host__ __device__ AABB() {
        min_t.x = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
        min_t.y = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
        min_t.z = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

        max_t.x = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
        max_t.y = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
        max_t.z = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
    };

    __host__ __device__ AABB(const vec3<T> &min_t, const vec3<T> &max_t)
        : min_t(min_t), max_t(max_t){};
    __host__ __device__ ~AABB(){};

    __host__ __device__ AABB(T min_t_x, T min_t_y, T min_t_z, T max_t_x, T max_t_y, T max_t_z) {
        min_t.x = min_t_x;
        min_t.y = min_t_y;
        min_t.z = min_t_z;
        max_t.x = max_t_x;
        max_t.y = max_t_y;
        max_t.z = max_t_z;
    }

    __host__ __device__ AABB<T> operator+(const AABB<T> &bbox2) const {
        return AABB<T>(
            min(this->min_t.x, bbox2.min_t.x), min(this->min_t.y, bbox2.min_t.y),
            min(this->min_t.z, bbox2.min_t.z), max(this->max_t.x, bbox2.max_t.x),
            max(this->max_t.y, bbox2.max_t.y), max(this->max_t.z, bbox2.max_t.z));
    };

    __host__ __device__ T distance(const vec3<T> point) const {};

    __host__ __device__ T operator*(const AABB<T> &bbox2) const {
        return (min(this->max_t.x, bbox2.max_t.x) -
                max(this->min_t.x, bbox2.min_t.x)) *
               (min(this->max_t.y, bbox2.max_t.y) -
                max(this->min_t.y, bbox2.min_t.y)) *
               (min(this->max_t.z, bbox2.max_t.z) -
                max(this->min_t.z, bbox2.min_t.z));
    };

    vec3<T> min_t;
    vec3<T> max_t;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const AABB<T> &x) {
    os << x.min_t << std::endl;
    os << x.max_t << std::endl;
    return os;
}

template <typename T>
struct MergeAABB {
   public:
    __host__ __device__ MergeAABB(){};

    // Create an operator Struct that will be used by thrust::reduce
    // to calculate the bounding box of the scene.
    __host__ __device__ AABB<T> operator()(const AABB<T> &bbox1,
                                           const AABB<T> &bbox2) {
        return bbox1 + bbox2;
    };
};

template <typename T>
__forceinline__
    __host__ __device__ T
    pointToAABBDistance(vec3<T> point, const AABB<T> &bbox) {
    T diff_x = point.x - clamp<T>(point.x, bbox.min_t.x, bbox.max_t.x);
    T diff_y = point.y - clamp<T>(point.y, bbox.min_t.y, bbox.max_t.y);
    T diff_z = point.z - clamp<T>(point.z, bbox.min_t.z, bbox.max_t.z);

    return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
}

template <typename T>
__forceinline__
    __host__ __device__ bool
    rayToAABBIntersect(vec3<T> point, vec3<T> direction, vec3<T> inverse, const AABB<T> &bbox) {
    // remove invalid small values to avoid division by zero
    // how the heck are there sooooo many undebuggable bugs?

    vec3<T> tmin = (bbox.min_t - point) * inverse;
    vec3<T> tmax = (bbox.max_t - point) * inverse;
    vec3<T> t1, t2;
    t1.x = min(tmin.x, tmax.x);
    t1.y = min(tmin.y, tmax.y);
    t1.z = min(tmin.z, tmax.z);

    t2.x = max(tmin.x, tmax.x);
    t2.y = max(tmin.y, tmax.y);
    t2.z = max(tmin.z, tmax.z);

    T near, far;
    near = max(max(t1.x, t1.y), t1.z);
    far = min(min(t2.x, t2.y), t2.z);

    return near < far;
}

template <typename T>
__forceinline__
    __host__ __device__ vec3<T>
    closest_point_on_segment(vec3<T> ray_origin, vec3<T> ray_direction, vec3<T> seg_start, vec3<T> seg_end, T &t) {
    vec3<T> seg_direction = seg_end - seg_start;
    vec3<T> origin_diff = ray_origin - seg_start;

    T a = dot(ray_direction, ray_direction);
    T b = dot(ray_direction, seg_direction);
    T c = dot(seg_direction, seg_direction);
    T d = dot(ray_direction, origin_diff);
    T e = dot(seg_direction, origin_diff);

    T denom = a * c - b * b;
    t = (b * e - c * d) / denom;

    // Clamp t to [0, 1] for segment
    t = t < 0 ? 0 : (t > 1 ? 1 : t);

    return seg_start + t * seg_direction;
}

template <typename T>
__forceinline__
    __host__ __device__ float
    rayToAABBDistance(vec3<T> ray_origin, vec3<T> ray_direction, vec3<T> ray_inverse, const AABB<T> &bbox) {
    // remove invalid small values to avoid division by zero
    // how the heck are there sooooo many undebuggable bugs?
    // TODO: precompute this and the ray direction inverse instead of everytime

    vec3<T> tmin = (bbox.min_t - ray_origin) * ray_inverse;
    vec3<T> tmax = (bbox.max_t - ray_origin) * ray_inverse;

    vec3<T> t1, t2;
    vec3<int> int1, int2;

    // clang-format off
    if (tmin.x < tmax.x) { int1.x = 0; t1.x = tmin.x; } // start will use bbox.min_t.x
    else { int1.x = 1; t1.x = tmax.x; }                 // start will use bbox.max_t.x
    if (tmin.y < tmax.y) { int1.y = 0; t1.y = tmin.y; }
    else { int1.y = 1; t1.y = tmax.y; }
    if (tmin.z < tmax.z) { int1.z = 0; t1.z = tmin.z; } 
    else { int1.z = 1; t1.z = tmax.z; }

    if (tmin.x > tmax.x) { int1.x = 0; t2.x = tmin.x; } // end will use bbox.min_t.x
    else { int1.x = 1; t2.x = tmax.x; }                 // end will use bbox.max_t.x
    if (tmin.y > tmax.y) { int1.y = 0; t2.y = tmin.y; }
    else { int1.y = 1; t2.y = tmax.y; }
    if (tmin.z > tmax.z) { int1.z = 0; t2.z = tmin.z; }
    else { int1.z = 1; t2.z = tmax.z; }
    // clang-format on

    T near, far;
    near = max(max(t1.x, t1.y), t1.z);
    far = min(min(t2.x, t2.y), t2.z);

    bool intersected = near < far;
    if (intersected) {
        T d2 = 0;
    } else {
        vec3<T> start, end;
        T t;
        start.x = bbox.max_t.x ? int1.x : bbox.min_t.x;
        start.y = bbox.max_t.y ? int1.y : bbox.min_t.y;
        start.z = bbox.max_t.z ? int1.z : bbox.min_t.z;
        end.x = bbox.min_t.x ? int2.x : bbox.max_t.x;
        end.y = bbox.min_t.y ? int2.y : bbox.max_t.y;
        end.z = bbox.min_t.z ? int2.z : bbox.max_t.z;
        vec3<T> point = closest_point_on_segment(point, ray_direction, start, end, t);
        T d2 = length_squared(ray_origin - point);
    }
    return d2;
}

#endif  // ifndef AABB_H
