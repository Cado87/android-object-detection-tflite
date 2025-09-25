package org.tensorflow.lite.examples.objectdetection.yolo

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/**
 * Utility class for YOLO post-processing including NMS (Non-Maximum Suppression)
 */
object YoloPostProcessor {
    
    /**
     * Process YOLO raw output and return filtered detections
     */
    fun processOutput(
        output: Array<FloatArray>,
        confidenceThreshold: Float,
        iouThreshold: Float,
        maxResults: Int,
        classNames: Array<String>,
        inputSize: Int = 640,
        originalWidth: Int,
        originalHeight: Int
    ): List<YoloDetection> {
        
        val detections = mutableListOf<YoloDetection>()
        
        // YOLO11n output format: [84, numDetections] where 84 = [x, y, w, h, 80 class scores]
        val numDetections = output[0].size // 8400 for 640x640, 2100 for 320x320
        val numClasses = classNames.size // 80
        val inputSizeFloat = inputSize.toFloat() // YOLO model input size (320 or 640)
        
        // Calculate scaling factors to map from square model input back to rectangular original image
        // The image was resized from originalWidth x originalHeight to inputSize x inputSize
        val scaleX = originalWidth.toFloat() / inputSizeFloat
        val scaleY = originalHeight.toFloat() / inputSizeFloat
        
        // Extract detections above confidence threshold
        for (i in 0 until numDetections) {
            // YOLO11n TFLite appears to output very small coordinate values that need to be scaled up
            // Based on logs showing values like 0.001288005, these seem to be normalized differently
            val rawX = output[0][i]
            val rawY = output[1][i]
            val rawW = output[2][i]
            val rawH = output[3][i]
            
            // Scale coordinates from model input space to original image pixel space
            val x = rawX * inputSizeFloat * scaleX
            val y = rawY * inputSizeFloat * scaleY
            val w = rawW * inputSizeFloat * scaleX
            val h = rawH * inputSizeFloat * scaleY
            
            // Find the class with highest confidence
            var maxScore = 0f
            var maxClass = 0
            
            for (j in 4 until (4 + numClasses)) {
                val score = output[j][i]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = j - 4
                }
            }
            
            // Apply confidence threshold
            if (maxScore >= confidenceThreshold && maxClass < classNames.size) {
                // Convert from center coordinates to corner coordinates in original image pixel space
                val left = x - w / 2f
                val top = y - h / 2f
                val right = x + w / 2f
                val bottom = y + h / 2f
                
                // Normalize to [0,1] range based on original image dimensions
                val normalizedLeft = (left / originalWidth).coerceIn(0f, 1f)
                val normalizedTop = (top / originalHeight).coerceIn(0f, 1f)
                val normalizedRight = (right / originalWidth).coerceIn(0f, 1f)
                val normalizedBottom = (bottom / originalHeight).coerceIn(0f, 1f)
                
                // Debug logging for coordinate transformation
                android.util.Log.d("YoloPostProcessor", 
                    "Detection: class=${classNames[maxClass]} conf=$maxScore " +
                    "raw_coords=($rawX, $rawY, $rawW, $rawH) " +
                    "scaled_to_original=($x, $y, $w, $h) " +
                    "bbox_pixels=($left, $top, $right, $bottom) " +
                    "bbox_normalized=($normalizedLeft, $normalizedTop, $normalizedRight, $normalizedBottom) " +
                    "scaleFactors=($scaleX, $scaleY)")
                
                val boundingBox = RectF(
                    normalizedLeft,
                    normalizedTop,
                    normalizedRight,
                    normalizedBottom
                )
                
                detections.add(
                    YoloDetection(
                        boundingBox = boundingBox,
                        confidence = maxScore,
                        classIndex = maxClass,
                        label = classNames[maxClass]
                    )
                )
            }
        }
        
        // Apply Non-Maximum Suppression
        val nmsResults = applyNMS(detections, iouThreshold)
        
        // Sort by confidence and limit results
        return nmsResults
            .sortedByDescending { it.confidence }
            .take(maxResults)
    }
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections
     */
    private fun applyNMS(detections: List<YoloDetection>, iouThreshold: Float): List<YoloDetection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<YoloDetection>()
        
        for (detection in sortedDetections) {
            var shouldSelect = true
            
            for (selected in selectedDetections) {
                val iou = calculateIoU(detection.boundingBox, selected.boundingBox)
                if (iou > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            
            if (shouldSelect) {
                selectedDetections.add(detection)
            }
        }
        
        return selectedDetections
    }
    
    /**
     * Calculate Intersection over Union (IoU) between two bounding boxes
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)
        
        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }
        
        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
}