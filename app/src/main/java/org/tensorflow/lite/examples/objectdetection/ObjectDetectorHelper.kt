/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.    fun clearObjectDetector() {
        synchronized(this) {
            try {
                // Shutdown the inference executor
                inferenceExecutor.shutdown()
                
                // Temporarily disabled Task API
                // objectDetector?.close()
                // objectDetector = null
                yoloDetector?.close()
                yoloDetector = null
                Log.d("ObjectDetectorHelper", "Detectors cleared successfully")
            } catch (e: Exception) {
                Log.e("ObjectDetectorHelper", "Error clearing detectors: ${e.message}", e)
            }
        }
    }ed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
// Temporarily disabled Task API imports
// import org.tensorflow.lite.task.core.BaseOptions
// import org.tensorflow.lite.task.vision.detector.Detection
// import org.tensorflow.lite.task.vision.detector.ObjectDetector
// import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.examples.objectdetection.yolo.YoloDetector
import org.tensorflow.lite.examples.objectdetection.yolo.YoloDetection
import org.tensorflow.lite.examples.objectdetection.yolo.YoloResult
import java.nio.ByteBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    // Temporarily disabled Task API
    // private var objectDetector: ObjectDetector? = null
    
    // YOLO detector for YOLO11n models
    private var yoloDetector: YoloDetector? = null
    
    // Track whether we're using YOLO or Task API - for now, force YOLO only
    private val isYoloModel: Boolean = true
        // get() = currentModel == MODEL_YOLO11N_FLOAT16 || currentModel == MODEL_YOLO11N_FLOAT32

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        synchronized(this) {
            try {
                // Temporarily disabled Task API
                // objectDetector?.close()
                // objectDetector = null
                yoloDetector?.close()
                yoloDetector = null
                Log.d("ObjectDetectorHelper", "Detectors cleared successfully")
            } catch (e: Exception) {
                Log.e("ObjectDetectorHelper", "Error clearing detectors: ${e.message}", e)
            }
        }
    }

    // Background execution
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "YOLOInference").apply {
            priority = Thread.NORM_PRIORITY - 1
        }
    }
    private val isInferenceRunning = AtomicBoolean(false)

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        // Clear existing detectors first to prevent conflicts
        clearObjectDetector()
        
        // Add a small delay to ensure cleanup is complete
        try {
            Thread.sleep(100)
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        
        // For now, always use YOLO
        setupYoloDetector()
    }
    
    private fun setupYoloDetector() {
        try {
            Log.d("ObjectDetectorHelper", "Setting up YOLO detector...")
            yoloDetector = YoloDetector(
                context = context,
                confidenceThreshold = threshold,
                maxResults = maxResults,
                numThreads = numThreads,
                useGpu = false  // Disable GPU for YOLO to prevent conflicts
            )
            
            val modelName = when (currentModel) {
                MODEL_YOLO11N_FLOAT16 -> "yolo11n_float16.tflite"
                MODEL_YOLO11N_FLOAT32 -> "yolo11n_float32.tflite"
                else -> "yolo11n_float32.tflite"
            }
            
            Log.d("ObjectDetectorHelper", "Loading YOLO model: $modelName")
            val success = yoloDetector?.loadModel(modelName, "metadata.yaml")
            if (success != true) {
                objectDetectorListener?.onError("Failed to load YOLO model: $modelName")
                yoloDetector?.close()
                yoloDetector = null
            } else {
                Log.d("ObjectDetectorHelper", "YOLO detector setup successful")
            }
        } catch (e: Exception) {
            objectDetectorListener?.onError("Error setting up YOLO detector: ${e.message}")
            Log.e("ObjectDetectorHelper", "YOLO setup error", e)
            yoloDetector?.close()
            yoloDetector = null
        }
    }
    
    /*
    // Temporarily disabled Task API methods
    private fun setupTaskApiDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                else -> "mobilenetv1.tflite"
            }

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("ObjectDetectorHelper", "TFLite failed to load model with error: " + e.message)
        }
    }
    */

    fun detect(image: Bitmap, imageRotation: Int) {
        // Skip if inference is already running to prevent backing up frames
        if (isInferenceRunning.get()) {
            Log.d("ObjectDetectorHelper", "Skipping frame - inference already running")
            return
        }
        
        try {
            // For now, always use YOLO
            Log.d("ObjectDetectorHelper", "Using YOLO detection path (YOLO-only mode)")
            detectWithYolo(image, imageRotation)
        } catch (e: Exception) {
            Log.e("ObjectDetectorHelper", "Error in detect method: ${e.message}", e)
            objectDetectorListener?.onError("Detection failed: ${e.message}")
        }
    }
    
    private fun detectWithYolo(image: Bitmap, imageRotation: Int) {
        // Skip if already processing
        if (!isInferenceRunning.compareAndSet(false, true)) {
            Log.d("ObjectDetectorHelper", "Detection already in progress, skipping frame")
            return
        }
        
        val detector = yoloDetector
        if (detector == null) {
            isInferenceRunning.set(false)
            Log.w("ObjectDetectorHelper", "YOLO detector is null, attempting to recreate...")
            setupObjectDetector()
            return
        }

        // Run inference on background thread to prevent ANR
        inferenceExecutor.execute {
            try {
                Log.d("ObjectDetectorHelper", "Running YOLO detection on background thread...")
                val result = detector.detect(image, imageRotation)
                
                // Post results back to main thread
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    try {
                        if (result != null) {
                            Log.d("ObjectDetectorHelper", "YOLO detection successful: ${result.detections.size} objects found")
                            // Convert YOLO detections to Task API format for compatibility
                            val simpleDetections = convertYoloToSimpleDetection(result.detections)
                            objectDetectorListener?.onResults(
                                simpleDetections.toMutableList(),
                                result.inferenceTime,
                                result.imageHeight,
                                result.imageWidth
                            )
                        } else {
                            Log.e("ObjectDetectorHelper", "YOLO detection returned null result")
                            objectDetectorListener?.onError("YOLO detection failed")
                        }
                    } finally {
                        isInferenceRunning.set(false)
                    }
                }
            } catch (e: Exception) {
                Log.e("ObjectDetectorHelper", "Error in YOLO detection: ${e.message}", e)
                // Post error back to main thread
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    try {
                        objectDetectorListener?.onError("YOLO detection error: ${e.message}")
                    } finally {
                        isInferenceRunning.set(false)
                    }
                }
            }
        }
    }
    
    /*
    // Temporarily disabled Task API detection method
    private fun detectWithTaskApi(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            Log.w("ObjectDetectorHelper", "Task API detector is null, attempting to recreate...")
            setupObjectDetector()
            return
        }

        try {
            // Inference time is the difference between the system time at the start and finish of the
            // process
            var inferenceTime = SystemClock.uptimeMillis()

            // Create preprocessor for the image.
            // See https://www.tensorflow.org/lite/inference_with_metadata/
            //            lite_support#imageprocessor_architecture
            val imageProcessor =
                ImageProcessor.Builder()
                    .add(Rot90Op(-imageRotation / 90))
                    .build()

            // Preprocess the image and convert it into a TensorImage for detection.
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

            val results = objectDetector?.detect(tensorImage)
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime
            
            Log.d("ObjectDetectorHelper", "Task API detection successful: ${results?.size ?: 0} objects found")
            objectDetectorListener?.onResults(
                results,
                inferenceTime,
                tensorImage.height,
                tensorImage.width)
        } catch (e: Exception) {
            Log.e("ObjectDetectorHelper", "Error in Task API detection: ${e.message}", e)
            objectDetectorListener?.onError("Detection error: ${e.message}")
        }
    }
    */
    
    /**
     * Convert YOLO detections to SimpleDetection for UI display
     */
    private fun convertYoloToSimpleDetection(yoloDetections: List<YoloDetection>): List<SimpleDetection> {
        return yoloDetections.map { yolo ->
            // Create a simple SimpleDetection object
            SimpleDetection.create(
                yolo.boundingBox,
                listOf(
                    SimpleDetection.Category.create(
                        yolo.label,
                        yolo.className,
                        yolo.confidence
                    )
                )
            )
        }
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: MutableList<SimpleDetection>?,
          inferenceTime: Long,
          imageHeight: Int,
          imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_YOLO11N_FLOAT16 = 4
        const val MODEL_YOLO11N_FLOAT32 = 5
    }
}
