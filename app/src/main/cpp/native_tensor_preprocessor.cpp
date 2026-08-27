#include <jni.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline int source_index(int ox, int oy, int width, int height, int degrees) {
    int sx, sy;
    switch (degrees) {
        case 0:   sx = ox;              sy = oy;                  break;
        case 90:  sx = oy;              sy = height - 1 - ox;    break;
        case 180: sx = width - 1 - ox;  sy = height - 1 - oy;    break;
        default:  sx = width - 1 - oy;  sy = ox;                  break;
    }
    return sy * width + sx;
}

inline float encoded(const jint* plane, int p, double gain, const jint* lut, int lut_max) {
    int index = static_cast<int>(static_cast<double>(plane[p]) * gain);
    index = std::max(0, std::min(index, lut_max));
    return static_cast<float>(lut[index]);
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_example_activeperception_NativeTensorPreprocessor_clearTensor(
        JNIEnv* env, jobject, jobject destination, jint float_offset, jint float_count,
        jfloat value) {
    auto* dst = static_cast<float*>(env->GetDirectBufferAddress(destination));
    if (!dst) return;
    dst += float_offset;
    const float32x4_t fill = vdupq_n_f32(value);
    int i = 0;
    for (; i + 16 <= float_count; i += 16) {
        vst1q_f32(dst + i, fill); vst1q_f32(dst + i + 4, fill);
        vst1q_f32(dst + i + 8, fill); vst1q_f32(dst + i + 12, fill);
    }
    for (; i + 4 <= float_count; i += 4) vst1q_f32(dst + i, fill);
    for (; i < float_count; ++i) dst[i] = value;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_activeperception_NativeTensorPreprocessor_fillOriginalSrgbStripe(
        JNIEnv* env, jobject,
        jintArray r_array, jintArray g_array, jintArray b_array,
        jint width, jint height, jint orientation, jdouble gain,
        jintArray lut_array, jobject destination, jint base_float,
        jint image_size, jint output_y_start, jint output_y_end) {
    auto* dst = static_cast<float*>(env->GetDirectBufferAddress(destination));
    if (!dst) return JNI_FALSE;
    auto* r = static_cast<jint*>(env->GetPrimitiveArrayCritical(r_array, nullptr));
    auto* g = static_cast<jint*>(env->GetPrimitiveArrayCritical(g_array, nullptr));
    auto* b = static_cast<jint*>(env->GetPrimitiveArrayCritical(b_array, nullptr));
    auto* lut = static_cast<jint*>(env->GetPrimitiveArrayCritical(lut_array, nullptr));
    if (!r || !g || !b || !lut) {
        if (r) env->ReleasePrimitiveArrayCritical(r_array, r, JNI_ABORT);
        if (g) env->ReleasePrimitiveArrayCritical(g_array, g, JNI_ABORT);
        if (b) env->ReleasePrimitiveArrayCritical(b_array, b, JNI_ABORT);
        if (lut) env->ReleasePrimitiveArrayCritical(lut_array, lut, JNI_ABORT);
        return JNI_FALSE;
    }

    const int degrees = ((orientation % 360) + 360) % 360;
    const int oriented_w = (degrees == 90 || degrees == 270) ? height : width;
    const int oriented_h = (degrees == 90 || degrees == 270) ? width : height;
    const float scale = std::min(static_cast<float>(image_size) / oriented_w,
                                 static_cast<float>(image_size) / oriented_h);
    const int nw = std::lround(oriented_w * scale);
    const int nh = std::lround(oriented_h * scale);
    const int pad_x = (image_size - nw) / 2;
    const int pad_y = (image_size - nh) / 2;
    const int lut_max = env->GetArrayLength(lut_array) - 1;
    dst += base_float;

    std::vector<int> x0(nw), x1(nw);
    std::vector<float> fx(nw);
    for (int dx = 0; dx < nw; ++dx) {
        const double sx = std::clamp((dx + 0.5) * oriented_w / nw - 0.5,
                                     0.0, static_cast<double>(oriented_w - 1));
        x0[dx] = static_cast<int>(std::floor(sx));
        x1[dx] = std::min(x0[dx] + 1, oriented_w - 1);
        fx[dx] = static_cast<float>(sx - x0[dx]);
    }

    const int begin = std::max(0, output_y_start);
    const int end = std::min(nh, output_y_end);
    alignas(16) float q00[3][4], q10[3][4], q01[3][4], q11[3][4];
    const jint* planes[3] = {r, g, b};
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    for (int dy = begin; dy < end; ++dy) {
        const double sy = std::clamp((dy + 0.5) * oriented_h / nh - 0.5,
                                     0.0, static_cast<double>(oriented_h - 1));
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = std::min(y0 + 1, oriented_h - 1);
        const float fy_scalar = static_cast<float>(sy - y0);
        const float32x4_t fyv = vdupq_n_f32(fy_scalar);
        int dx = 0;
        for (; dx + 4 <= nw; dx += 4) {
            for (int lane = 0; lane < 4; ++lane) {
                const int x = dx + lane;
                const int p00 = source_index(x0[x], y0, width, height, degrees);
                const int p10 = source_index(x1[x], y0, width, height, degrees);
                const int p01 = source_index(x0[x], y1, width, height, degrees);
                const int p11 = source_index(x1[x], y1, width, height, degrees);
                for (int c = 0; c < 3; ++c) {
                    q00[c][lane] = encoded(planes[c], p00, gain, lut, lut_max);
                    q10[c][lane] = encoded(planes[c], p10, gain, lut, lut_max);
                    q01[c][lane] = encoded(planes[c], p01, gain, lut, lut_max);
                    q11[c][lane] = encoded(planes[c], p11, gain, lut, lut_max);
                }
            }
            const float32x4_t fxv = vld1q_f32(fx.data() + dx);
            float32x4x3_t rgb;
            for (int c = 0; c < 3; ++c) {
                const float32x4_t a = vld1q_f32(q00[c]);
                const float32x4_t top = vfmaq_f32(a, vsubq_f32(vld1q_f32(q10[c]), a), fxv);
                const float32x4_t d = vld1q_f32(q01[c]);
                const float32x4_t bottom = vfmaq_f32(d, vsubq_f32(vld1q_f32(q11[c]), d), fxv);
                rgb.val[c] = vmulq_f32(vfmaq_f32(top, vsubq_f32(bottom, top), fyv), inv255);
            }
            float* out = dst + ((dy + pad_y) * image_size + dx + pad_x) * 3;
            vst3q_f32(out, rgb);
        }
        for (; dx < nw; ++dx) {
            const int p00 = source_index(x0[dx], y0, width, height, degrees);
            const int p10 = source_index(x1[dx], y0, width, height, degrees);
            const int p01 = source_index(x0[dx], y1, width, height, degrees);
            const int p11 = source_index(x1[dx], y1, width, height, degrees);
            float* out = dst + ((dy + pad_y) * image_size + dx + pad_x) * 3;
            for (int c = 0; c < 3; ++c) {
                const float top = encoded(planes[c], p00, gain, lut, lut_max) * (1 - fx[dx])
                                + encoded(planes[c], p10, gain, lut, lut_max) * fx[dx];
                const float bottom = encoded(planes[c], p01, gain, lut, lut_max) * (1 - fx[dx])
                                   + encoded(planes[c], p11, gain, lut, lut_max) * fx[dx];
                out[c] = (top * (1 - fy_scalar) + bottom * fy_scalar) / 255.0f;
            }
        }
    }

    env->ReleasePrimitiveArrayCritical(lut_array, lut, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(b_array, b, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(g_array, g, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(r_array, r, JNI_ABORT);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_activeperception_NativeTensorPreprocessor_fillFusedBayerSrgbRowStripe(
        JNIEnv* env, jobject, jobjectArray frame_arrays, jint n_sum,
        jint bayer_width, jint bayer_height, jstring cfa_string,
        jint orientation, jdoubleArray gains_array, jintArray bases_array,
        jintArray lut_array, jobject destination, jint image_size,
        jint output_y_start, jint output_y_end) {
    auto* dst = static_cast<float*>(env->GetDirectBufferAddress(destination));
    if (!dst || !frame_arrays || !gains_array || !bases_array || !lut_array ||
        !cfa_string || n_sum <= 0) return JNI_FALSE;
    const int available = env->GetArrayLength(frame_arrays);
    const int used = std::min(static_cast<int>(n_sum), available);
    const int gain_count = env->GetArrayLength(gains_array);
    const int base_count = env->GetArrayLength(bases_array);
    const int lut_length = env->GetArrayLength(lut_array);
    if (used <= 0 || gain_count <= 0 || base_count != gain_count || lut_length <= 0) {
        return JNI_FALSE;
    }

    std::vector<jintArray> frame_refs(used);
    std::vector<jint*> frames(used, nullptr);
    bool ok = true;
    // Resolve every JNI object/length before the first critical pin. CheckJNI correctly
    // rejects GetObjectArrayElement/GetArrayLength while a primitive array is pinned.
    for (int k = 0; k < used; ++k) {
        frame_refs[k] = static_cast<jintArray>(env->GetObjectArrayElement(frame_arrays, k));
        if (!frame_refs[k] || env->GetArrayLength(frame_refs[k]) < bayer_width * bayer_height) {
            ok = false; break;
        }
    }
    const char* cfa = ok ? env->GetStringUTFChars(cfa_string, nullptr) : nullptr;
    if (!cfa) ok = false;
    for (int k = 0; ok && k < used; ++k) {
        frames[k] = static_cast<jint*>(env->GetPrimitiveArrayCritical(frame_refs[k], nullptr));
        if (!frames[k]) ok = false;
    }
    auto* gains = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(gains_array, nullptr));
    auto* bases = static_cast<jint*>(env->GetPrimitiveArrayCritical(bases_array, nullptr));
    auto* lut = static_cast<jint*>(env->GetPrimitiveArrayCritical(lut_array, nullptr));
    if (!gains || !bases || !lut || !cfa) ok = false;

    if (ok) {
        int r_site = -1, b_site = -1, g_sites[2] = {-1, -1}, gn = 0;
        for (int i = 0; i < 4; ++i) {
            if (cfa[i] == 'R') r_site = i;
            else if (cfa[i] == 'B') b_site = i;
            else if (gn < 2) g_sites[gn++] = i;
        }
        ok = r_site >= 0 && b_site >= 0 && gn == 2;
        if (ok) {
            const int width = bayer_width / 2;
            const int height = bayer_height / 2;
            const int degrees = ((orientation % 360) + 360) % 360;
            const int oriented_w = (degrees == 90 || degrees == 270) ? height : width;
            const int oriented_h = (degrees == 90 || degrees == 270) ? width : height;
            const float scale = std::min(static_cast<float>(image_size) / oriented_w,
                                         static_cast<float>(image_size) / oriented_h);
            const int nw = std::lround(oriented_w * scale);
            const int nh = std::lround(oriented_h * scale);
            const int pad_x = (image_size - nw) / 2;
            const int pad_y = (image_size - nh) / 2;
            const int lut_max = lut_length - 1;
            std::vector<int> x0(nw), x1(nw);
            std::vector<float> fx(nw);
            for (int dx = 0; dx < nw; ++dx) {
                const double sx = std::clamp((dx + 0.5) * oriented_w / nw - 0.5,
                                             0.0, static_cast<double>(oriented_w - 1));
                x0[dx] = static_cast<int>(std::floor(sx));
                x1[dx] = std::min(x0[dx] + 1, oriented_w - 1);
                fx[dx] = static_cast<float>(sx - x0[dx]);
            }
            auto demosaic = [&](int oriented_x, int oriented_y, int out_rgb[3]) {
                const int p = source_index(oriented_x, oriented_y, width, height, degrees);
                const int bx = p % width;
                const int by = p / width;
                const int base = (by * 2) * bayer_width + bx * 2;
                const int offsets[4] = {0, 1, bayer_width, bayer_width + 1};
                int sites[4] = {0, 0, 0, 0};
                for (int k = 0; k < used; ++k) {
                    const jint* frame = frames[k];
                    sites[0] += frame[base + offsets[0]];
                    sites[1] += frame[base + offsets[1]];
                    sites[2] += frame[base + offsets[2]];
                    sites[3] += frame[base + offsets[3]];
                }
                out_rgb[0] = sites[r_site];
                out_rgb[1] = (sites[g_sites[0]] + sites[g_sites[1]]) / 2;
                out_rgb[2] = sites[b_site];
            };
            const int begin = std::max(0, static_cast<int>(output_y_start));
            const int end = std::min(nh, static_cast<int>(output_y_end));
            for (int dy = begin; dy < end; ++dy) {
                const double sy = std::clamp((dy + 0.5) * oriented_h / nh - 0.5,
                                             0.0, static_cast<double>(oriented_h - 1));
                const int y0 = static_cast<int>(std::floor(sy));
                const int y1 = std::min(y0 + 1, oriented_h - 1);
                const float fy = static_cast<float>(sy - y0);
                for (int dx = 0; dx < nw; ++dx) {
                    int q00[3], q10[3], q01[3], q11[3];
                    demosaic(x0[dx], y0, q00); demosaic(x1[dx], y0, q10);
                    demosaic(x0[dx], y1, q01); demosaic(x1[dx], y1, q11);
                    for (int lane = 0; lane < gain_count; ++lane) {
                        float* out = dst + bases[lane] +
                            ((dy + pad_y) * image_size + dx + pad_x) * 3;
                        const double gain = gains[lane];
                        for (int c = 0; c < 3; ++c) {
                            auto enc = [&](int value) -> float {
                                int index = static_cast<int>(value * gain);
                                index = std::max(0, std::min(index, lut_max));
                                return static_cast<float>(lut[index]);
                            };
                            const float top = enc(q00[c]) * (1.0f - fx[dx]) +
                                              enc(q10[c]) * fx[dx];
                            const float bottom = enc(q01[c]) * (1.0f - fx[dx]) +
                                                 enc(q11[c]) * fx[dx];
                            out[c] = (top * (1.0f - fy) + bottom * fy) / 255.0f;
                        }
                    }
                }
            }
        }
    }

    if (lut) env->ReleasePrimitiveArrayCritical(lut_array, lut, JNI_ABORT);
    if (bases) env->ReleasePrimitiveArrayCritical(bases_array, bases, JNI_ABORT);
    if (gains) env->ReleasePrimitiveArrayCritical(gains_array, gains, JNI_ABORT);
    for (int k = 0; k < used; ++k) {
        if (frames[k]) env->ReleasePrimitiveArrayCritical(frame_refs[k], frames[k], JNI_ABORT);
    }
    if (cfa) env->ReleaseStringUTFChars(cfa_string, cfa);
    for (int k = 0; k < used; ++k) {
        if (frame_refs[k]) env->DeleteLocalRef(frame_refs[k]);
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_activeperception_NativeTensorPreprocessor_decodeSubsampledBayer(
        JNIEnv* env, jobject, jobject source, jint source_byte_offset,
        jint source_width, jint source_height, jint row_stride, jint pixel_stride,
        jintArray black_array, jintArray destination_array) {
    auto* base = static_cast<const uint8_t*>(env->GetDirectBufferAddress(source));
    if (!base || !black_array || !destination_array) return JNI_FALSE;
    auto* black = static_cast<jint*>(env->GetPrimitiveArrayCritical(black_array, nullptr));
    auto* destination = static_cast<jint*>(
            env->GetPrimitiveArrayCritical(destination_array, nullptr));
    if (!black || !destination) {
        if (black) env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
        if (destination) env->ReleasePrimitiveArrayCritical(
                destination_array, destination, JNI_ABORT);
        return JNI_FALSE;
    }

    base += source_byte_offset;
    const int output_width = (source_width / 2) & ~1;
    const int output_height = (source_height / 2) & ~1;
    if (env->GetArrayLength(destination_array) < output_width * output_height) {
        env->ReleasePrimitiveArrayCritical(destination_array, destination, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
        return JNI_FALSE;
    }

    for (int oy = 0; oy < output_height; ++oy) {
        const int iy = (oy & ~1) * 2 + (oy & 1);
        const uint8_t* row_bytes = base + static_cast<size_t>(iy) * row_stride;
        jint* out = destination + static_cast<size_t>(oy) * output_width;
        const int channel_row = (iy & 1) * 2;
        int ox = 0;

        // RayNeo RAW_SENSOR uses little-endian uint16 pixels. Deinterleave each 32-pixel
        // source group with vld4: val[0]/val[1] are exactly CFA-preserving positions
        // 0,1,4,5,...,28,29 selected by the Kotlin reference implementation.
        if (pixel_stride == 2) {
            const auto* row = reinterpret_cast<const uint16_t*>(row_bytes);
            const uint16x8_t b0 = vdupq_n_u16(static_cast<uint16_t>(black[channel_row]));
            const uint16x8_t b1 = vdupq_n_u16(static_cast<uint16_t>(black[channel_row + 1]));
            for (; ox + 16 <= output_width; ox += 16) {
                const int ix = (ox / 2) * 4;
                const uint16x8x4_t pixels = vld4q_u16(row + ix);
                const uint16x8_t even = vqsubq_u16(pixels.val[0], b0);
                const uint16x8_t odd = vqsubq_u16(pixels.val[1], b1);
                uint32x4x2_t low = { vmovl_u16(vget_low_u16(even)),
                                     vmovl_u16(vget_low_u16(odd)) };
                uint32x4x2_t high = { vmovl_u16(vget_high_u16(even)),
                                      vmovl_u16(vget_high_u16(odd)) };
                vst2q_u32(reinterpret_cast<uint32_t*>(out + ox), low);
                vst2q_u32(reinterpret_cast<uint32_t*>(out + ox + 8), high);
            }
        }
        for (; ox < output_width; ++ox) {
            const int ix = (ox & ~1) * 2 + (ox & 1);
            const uint8_t* pixel = row_bytes + static_cast<size_t>(ix) * pixel_stride;
            const int raw = static_cast<int>(pixel[0]) |
                            (static_cast<int>(pixel[1]) << 8);
            const int value = raw - black[channel_row + (ix & 1)];
            out[ox] = std::max(0, value);
        }
    }

    env->ReleasePrimitiveArrayCritical(destination_array, destination, 0);
    env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_activeperception_NativeTensorPreprocessor_decodeSubsampledRaw10(
        JNIEnv* env, jobject, jobject source, jint source_byte_offset,
        jint source_width, jint source_height, jint row_stride,
        jintArray black_array, jintArray destination_array) {
    auto* base = static_cast<const uint8_t*>(env->GetDirectBufferAddress(source));
    if (!base || !black_array || !destination_array || (source_width & 3) != 0) {
        return JNI_FALSE;
    }
    auto* black = static_cast<jint*>(env->GetPrimitiveArrayCritical(black_array, nullptr));
    auto* destination = static_cast<jint*>(
            env->GetPrimitiveArrayCritical(destination_array, nullptr));
    if (!black || !destination) {
        if (black) env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
        if (destination) env->ReleasePrimitiveArrayCritical(
                destination_array, destination, JNI_ABORT);
        return JNI_FALSE;
    }

    base += source_byte_offset;
    const int output_width = (source_width / 2) & ~1;
    const int output_height = (source_height / 2) & ~1;
    if (env->GetArrayLength(destination_array) < output_width * output_height) {
        env->ReleasePrimitiveArrayCritical(destination_array, destination, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
        return JNI_FALSE;
    }

    // Android RAW10 stores four 10-bit pixels in five bytes. Bytes 0..3 contain each
    // pixel's upper eight bits and byte 4 contains the four 2-bit tails. The SoS 2x
    // subsampling keeps pixels 0 and 1 from every group, so no unpacked full-size buffer
    // is ever allocated.
    for (int oy = 0; oy < output_height; ++oy) {
        const int iy = (oy & ~1) * 2 + (oy & 1);
        const uint8_t* row = base + static_cast<size_t>(iy) * row_stride;
        jint* out = destination + static_cast<size_t>(oy) * output_width;
        const int channel_row = (iy & 1) * 2;
        const int groups = output_width / 2;
        for (int group = 0; group < groups; ++group) {
            const uint8_t* packed = row + static_cast<size_t>(group) * 5;
            const int tails = packed[4];
            const int p0 = (static_cast<int>(packed[0]) << 2) | (tails & 0x3);
            const int p1 = (static_cast<int>(packed[1]) << 2) | ((tails >> 2) & 0x3);
            out[group * 2] = std::max(0, p0 - black[channel_row]);
            out[group * 2 + 1] = std::max(0, p1 - black[channel_row + 1]);
        }
    }

    env->ReleasePrimitiveArrayCritical(destination_array, destination, 0);
    env->ReleasePrimitiveArrayCritical(black_array, black, JNI_ABORT);
    return JNI_TRUE;
}
