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
        classNames: Array<String>
    ): List<YoloDetection> {
        
        val detections = mutableListOf<YoloDetection>()
        
        // YOLO11n output format: [84, 8400] where 84 = [x, y, w, h, 80 class scores]
        val numDetections = output[0].size // 8400
        val numClasses = classNames.size // 80
        
        // Extract detections above confidence threshold
        for (i in 0 until numDetections) {
            val x = output[0][i]
            val y = output[1][i]
            val w = output[2][i]
            val h = output[3][i]
            
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
                val left = x - w / 2f
                val top = y - h / 2f
                val right = x + w / 2f
                val bottom = y + h / 2f
                
                val boundingBox = RectF(
                    left.coerceIn(0f, 1f),
                    top.coerceIn(0f, 1f),
                    right.coerceIn(0f, 1f),
                    bottom.coerceIn(0f, 1f)
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