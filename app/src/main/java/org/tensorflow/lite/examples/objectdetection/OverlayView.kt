/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.examples.objectdetection.SimpleDetection

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<SimpleDetection> = LinkedList<SimpleDetection>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        Log.d("OverlayView", "Drawing overlay with ${results.size} results")
        for (result in results) {
            val boundingBox = result.boundingBox

            // Check if coordinates are normalized (YOLO) or pixel-based (Task API)
            val isNormalized = results.isNotEmpty() && 
                results.all { it.boundingBox.right <= 1.0f && it.boundingBox.bottom <= 1.0f }
            
            val top: Float
            val bottom: Float
            val left: Float
            val right: Float
            
            if (isNormalized) {
                // TEMPORARY: Simple direct mapping to debug coordinate issues
                // Map normalized coordinates directly to full view dimensions
                left = boundingBox.left * width
                top = boundingBox.top * height
                right = boundingBox.right * width
                bottom = boundingBox.bottom * height
                
                Log.d("OverlayView", "TEMP: Direct mapping - normalized: [${boundingBox.left}, ${boundingBox.top}, ${boundingBox.right}, ${boundingBox.bottom}]")
                Log.d("OverlayView", "TEMP: Direct mapping - screen: [${left}, ${top}, ${right}, ${bottom}]")
                Log.d("OverlayView", "TEMP: View dimensions: ${width}x${height}, Image dimensions: ${imageWidth}x${imageHeight}")
            } else {
                // Use traditional scaling for Task API pixel coordinates
                left = boundingBox.left * scaleFactor
                top = boundingBox.top * scaleFactor
                right = boundingBox.right * scaleFactor
                bottom = boundingBox.bottom * scaleFactor
            }

            Log.d("OverlayView", "Drawing box: left=$left, top=$top, right=$right, bottom=$bottom")

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                result.categories[0].label + " " +
                        String.format("%.2f", result.categories[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
      detectionResults: MutableList<SimpleDetection>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        Log.d("OverlayView", "Setting ${detectionResults.size} results, imageSize: ${imageWidth}x${imageHeight}, viewSize: ${width}x${height}")
        results = detectionResults
        
        // Store image dimensions for use in draw()
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight

        // For YOLO models, coordinates are already normalized [0,1], so we scale directly by view dimensions
        // For Task API models, coordinates are in image pixel space, so we need the traditional scale factor
        
        // Check if coordinates are normalized (YOLO) or pixel-based (Task API)
        // YOLO coordinates are typically in [0,1] range, while Task API coordinates are larger
        val isNormalized = detectionResults.isNotEmpty() && 
            detectionResults.all { it.boundingBox.right <= 1.0f && it.boundingBox.bottom <= 1.0f }
        
        if (isNormalized) {
            // YOLO case: coordinates are normalized (0-1), no additional scaling needed
            scaleFactor = 1.0f
            Log.d("OverlayView", "Using normalized coordinates (YOLO), direct mapping to view")
        } else {
            // Task API case: coordinates are in image pixel space, need scaling
            scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
            Log.d("OverlayView", "Using pixel coordinates (Task API), scale factor: $scaleFactor")
        }
        
        results.forEachIndexed { index, result ->
            val bb = result.boundingBox
            Log.d("OverlayView", "Result $index: label=${result.categories[0].label}, score=${result.categories[0].score}, box=[${bb.left}, ${bb.top}, ${bb.right}, ${bb.bottom}]")
        }
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
