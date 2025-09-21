package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF

/**
 * Simple detection result class to replace Task API Detection during YOLO-only testing
 */
class SimpleDetection private constructor(
    val boundingBox: RectF,
    val categories: List<Category>
) {
    companion object {
        fun create(boundingBox: RectF, categories: List<Category>): SimpleDetection {
            return SimpleDetection(boundingBox, categories)
        }
    }
    
    /**
     * Simple category class to replace Task API Category
     */
    class Category private constructor(
        val label: String,
        val displayName: String,
        val score: Float
    ) {
        companion object {
            fun create(label: String, displayName: String, score: Float): Category {
                return Category(label, displayName, score)
            }
        }
    }
}