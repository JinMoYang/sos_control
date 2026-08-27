package com.example.activeperception

import java.nio.ByteBuffer

/** EXP2.1 native arm. Production paths never call this object. */
object NativeTensorPreprocessor {
    init { System.loadLibrary("sos_tensor") }

    external fun clearTensor(
        destination: ByteBuffer, floatOffset: Int, floatCount: Int, value: Float
    )

    /** Writes one horizontal content stripe into an already padded HWC float tensor. */
    external fun fillOriginalSrgbStripe(
        r: IntArray, g: IntArray, b: IntArray,
        width: Int, height: Int, orientation: Int, gain: Double,
        srgbLut: IntArray, destination: ByteBuffer, baseFloat: Int,
        imageSize: Int, outputYStart: Int, outputYEnd: Int
    ): Boolean

    /** Fuses first-N Bayer sum, 2x2 demosaic, digital gain, sRGB and letterbox resize. */
    external fun fillFusedBayerSrgbRowStripe(
        frames: Array<IntArray>, nSum: Int,
        bayerWidth: Int, bayerHeight: Int, cfaPattern: String,
        orientation: Int, gains: DoubleArray, baseFloats: IntArray,
        srgbLut: IntArray, destination: ByteBuffer,
        imageSize: Int, outputYStart: Int, outputYEnd: Int
    ): Boolean

    /** EXP5.1.1: CFA-preserving 2x RAW decode + per-channel black subtraction. */
    external fun decodeSubsampledBayer(
        source: ByteBuffer, sourceByteOffset: Int,
        sourceWidth: Int, sourceHeight: Int,
        rowStrideBytes: Int, pixelStrideBytes: Int,
        blackLevels: IntArray, destination: IntArray
    ): Boolean

    /** EXP5.3: Android RAW10 packed decode with the identical CFA-preserving sampling. */
    external fun decodeSubsampledRaw10(
        source: ByteBuffer, sourceByteOffset: Int,
        sourceWidth: Int, sourceHeight: Int,
        rowStrideBytes: Int, blackLevels: IntArray,
        destination: IntArray
    ): Boolean
}
