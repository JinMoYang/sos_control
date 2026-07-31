package com.example.activeperception

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** Detection-box overlay drawn on top of the preview ImageView. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 36f
        style = Paint.Style.FILL
    }

    private val textBgPaint = Paint().apply {
        color = Color.BLACK
        alpha = 160
        style = Paint.Style.FILL
    }

    private var drawData: List<DrawInfo> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1

    /** Reused across boxes so onDraw allocates nothing. */
    private val scratch = RectF()

    fun setResults(results: List<DrawInfo>, imgW: Int, imgH: Int) {
        this.drawData = results
        this.imageWidth = imgW
        this.imageHeight = imgH
        postInvalidate()
    }

    fun clear() {
        this.drawData = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (drawData.isEmpty()) return

        // Must match the ImageView's scaleType="fitCenter": one uniform scale, image centred,
        // shorter axis padded. Scaling each axis independently would stretch the boxes
        // whenever the view aspect differs from the bitmap's, which it always does in portrait.
        val scale = minOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val dispW = imageWidth * scale
        val dispH = imageHeight * scale
        val offsetX = (width - dispW) / 2f
        val offsetY = (height - dispH) / 2f

        for (item in drawData) {
            val rect = item.rect
            val left = offsetX + rect.left * scale
            val top = offsetY + rect.top * scale
            val right = offsetX + rect.right * scale
            val bottom = offsetY + rect.bottom * scale

            scratch.set(left, top, right, bottom)
            canvas.drawRect(scratch, boxPaint)

            if (item.text.isNotEmpty()) {
                val textWidth = textPaint.measureText(item.text)
                val ts = textPaint.textSize
                canvas.drawRect(left, top - ts - 6, left + textWidth + 12, top, textBgPaint)
                canvas.drawText(item.text, left + 6, top - 6, textPaint)
            }
        }
    }

    data class DrawInfo(val rect: Rect, val text: String)
}
