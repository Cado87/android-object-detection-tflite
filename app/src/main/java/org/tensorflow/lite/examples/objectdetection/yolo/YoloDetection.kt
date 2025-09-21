package org.tensorflow.lite.examples.objectdetection.yolo

import android.graphics.RectF

/**
 * Represents a detected object with bounding box and classification
 */
data class YoloDetection(
    val boundingBox: RectF,
    val confidence: Float,
    val classIndex: Int,
    val label: String
) {
    // Add className property for compatibility with existing code
    val className: String get() = label
}