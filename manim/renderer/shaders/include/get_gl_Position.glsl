// Assumes the following uniforms exist in the surrounding context:
// uniform vec2 frame_shape;
// uniform float focal_distance;
// uniform float is_fixed_in_frame;
// uniform float z_shift;
const vec2 DEFAULT_FRAME_SHAPE = vec2(8.0 * 16.0 / 9.0, 8.0);

float perspective_scale_factor(float z, float focal_distance)
{
    return focal_distance / (focal_distance - z);
}

vec4 get_gl_Position(vec3 point)
{
    vec4 result = vec4(point, 1.0);
    result = u_projection_view_matrix * result;
    result.xyzw /= result.w;
    // if (!bool(is_fixed_in_frame))
    // {
    //     result.x *= 2.0 / frame_shape.x;
    //     result.y *= 2.0 / frame_shape.y;
    //     float psf = perspective_scale_factor(result.z, focal_distance);
    //     result.xyz *= psf;
    //     // TODO, what's the better way to do this?
    //     // This is to keep vertices too far out of frame from getting cut.
    //     // TODO This will be done by the clipping plane in the future with the transformation matrix
    //     // result.z += z_shift;
    //     result.z *= (1 / 100.0);
    // }
    // else
    // {
    //     result.x *= 2.0 / DEFAULT_FRAME_SHAPE.x;
    //     result.y *= 2.0 / DEFAULT_FRAME_SHAPE.y;
    // }
    // result.z *= -1;
    return result;
}
