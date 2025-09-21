package org.tensorflow.lite.examples.objectdetection.yolo

/**
 * Data class representing the complete result from YOLO inference
 */
data class YoloResult(
    val detections: List<YoloDetection>,
    val inferenceTime: Long,
    val imageWidth: Int,
    val imageHeight: Int
)