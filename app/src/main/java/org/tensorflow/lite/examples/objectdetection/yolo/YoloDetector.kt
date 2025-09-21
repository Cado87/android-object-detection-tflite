package org.tensorflow.lite.examples.objectdetection.yolo

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.yaml.snakeyaml.Yaml
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * YOLO11n detector implementation using TensorFlow Lite Interpreter
 */
class YoloDetector(
    private val context: Context,
    var confidenceThreshold: Float = 0.5f,
    var iouThreshold: Float = 0.3f,
    var maxResults: Int = 3,
    var numThreads: Int = 2,
    var useGpu: Boolean = false
) {
    // Thread safety
    private val detectionLock = Any()
    @Volatile private var isDetecting = false
    
    private var interpreter: Interpreter? = null
    private var classNames: Array<String> = emptyArray()
    private var inputSize = 640 // Default YOLO input size
    
    companion object {
        private const val TAG = "YoloDetector"
        private const val BATCH_SIZE = 1
        private const val PIXEL_SIZE = 3 // RGB
        private const val NUM_BYTES_PER_CHANNEL = 4 // Float32
    }
    
    /**
     * Load YOLO model and metadata
     */
    fun loadModel(modelPath: String, metadataPath: String): Boolean {
        return try {
            Log.d(TAG, "Loading YOLO model: $modelPath")
            
            // Load metadata first to get class names and input size
            Log.d(TAG, "Loading metadata: $metadataPath")
            loadMetadata(metadataPath)
            
            // Load the TensorFlow Lite model
            Log.d(TAG, "Loading model file...")
            val modelBuffer = loadModelFile(modelPath)
            
            // Configure interpreter options
            Log.d(TAG, "Configuring interpreter options...")
            val options = Interpreter.Options().apply {
                setNumThreads(numThreads)
                
                if (useGpu && CompatibilityList().isDelegateSupportedOnThisDevice) {
                    try {
                        val delegateOptions = CompatibilityList().bestOptionsForThisDevice
                        val gpuDelegate = GpuDelegate(delegateOptions.setQuantizedModelsAllowed(true))
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU delegate enabled")
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to enable GPU delegate: ${e.message}")
                    }
                } else {
                    Log.d(TAG, "Using CPU inference")
                }
            }
            
            Log.d(TAG, "Creating interpreter...")
            interpreter = Interpreter(modelBuffer, options)
            
            // Verify input/output shapes
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            
            Log.d(TAG, "Model loaded successfully")
            Log.d(TAG, "Input shape: ${inputShape?.contentToString()}")
            Log.d(TAG, "Output shape: ${outputShape?.contentToString()}")
            Log.d(TAG, "Number of classes: ${classNames.size}")
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
            false
        }
    }
    
    /**
     * Run inference on the input image
     */
    fun detect(image: Bitmap, imageRotation: Int): YoloResult? {
        // Prevent concurrent access to the interpreter
        synchronized(detectionLock) {
            if (isDetecting) {
                Log.w(TAG, "Detection already in progress, skipping frame")
                return null
            }
            isDetecting = true
        }
        
        try {
            return detectInternal(image, imageRotation)
        } finally {
            synchronized(detectionLock) {
                isDetecting = false
            }
        }
    }
    
    /**
     * Internal detection method - protected by synchronization
     */
    private fun detectInternal(image: Bitmap, imageRotation: Int): YoloResult? {
        val interpreter = this.interpreter ?: return null
        
        // Basic input validation
        if (image.isRecycled) {
            Log.e(TAG, "Cannot process recycled bitmap")
            return null
        }
        
        if (classNames.isEmpty()) {
            Log.e(TAG, "No class names loaded")
            return null
        }
        
        val startTime = SystemClock.uptimeMillis()
        
        try {
            Log.d(TAG, "Starting YOLO inference for image ${image.width}x${image.height}")
            
            // Preprocess image
            val preprocessedImage = preprocessImage(image, imageRotation)
            Log.d(TAG, "Preprocessed image to ${preprocessedImage.width}x${preprocessedImage.height}")
            
            // Prepare input buffer with explicit size calculation
            val inputBufferSize = BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE * NUM_BYTES_PER_CHANNEL
            Log.d(TAG, "Allocating input buffer of size: $inputBufferSize bytes")
            
            val inputBuffer = ByteBuffer.allocateDirect(inputBufferSize).apply {
                order(ByteOrder.nativeOrder())
                rewind()
            }
            
            // Convert preprocessed image to input buffer
            imageToBuffer(preprocessedImage, inputBuffer)
            Log.d(TAG, "Converted image to input buffer")
            
            // Prepare output buffer - YOLO11n outputs [1, 84, 8400]
            val outputBuffer = Array(1) { Array(84) { FloatArray(8400) } }
            
            // Run inference
            Log.d(TAG, "Running inference...")
            
            // Ensure interpreter is still valid before running inference
            val currentInterpreter = this.interpreter
            if (currentInterpreter == null) {
                Log.e(TAG, "Interpreter became null during detection")
                return null
            }
            
            try {
                currentInterpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer),
                    mapOf(0 to outputBuffer)
                )
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Interpreter in invalid state: ${e.message}")
                return null
            } catch (e: RuntimeException) {
                Log.e(TAG, "Runtime error during inference: ${e.message}")
                return null
            }
            
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            Log.d(TAG, "Inference completed in ${inferenceTime}ms")
            
            // Post-process results
            val detections = YoloPostProcessor.processOutput(
                output = outputBuffer[0],
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold,
                maxResults = maxResults,
                classNames = classNames
            )
            
            Log.d(TAG, "Found ${detections.size} detections")
            
            // Scale bounding boxes back to original image dimensions
            val scaledDetections = scaleDetections(detections, image.width, image.height)
            
            return YoloResult(
                detections = scaledDetections,
                inferenceTime = inferenceTime,
                imageWidth = image.width,
                imageHeight = image.height
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}", e)
            return null
        }
    }
    
    /**
     * Preprocess input image for YOLO
     */
    private fun preprocessImage(image: Bitmap, imageRotation: Int): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        return imageProcessor.process(TensorImage.fromBitmap(image))
    }
    
    /**
     * Convert TensorImage to ByteBuffer for model input
     */
    private fun imageToBuffer(tensorImage: TensorImage, buffer: ByteBuffer) {
        val bitmap = tensorImage.bitmap
        val intValues = IntArray(inputSize * inputSize)
        
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[pixel++]
                
                // Normalize to [0, 1] and put in buffer (RGB order)
                buffer.putFloat(((pixelValue shr 16) and 0xFF) / 255f) // R
                buffer.putFloat(((pixelValue shr 8) and 0xFF) / 255f)  // G
                buffer.putFloat((pixelValue and 0xFF) / 255f)          // B
            }
        }
    }
    
    /**
     * Scale detections from normalized coordinates to image coordinates
     */
    private fun scaleDetections(detections: List<YoloDetection>, imageWidth: Int, imageHeight: Int): List<YoloDetection> {
        return detections.map { detection ->
            val scaledBox = android.graphics.RectF(
                detection.boundingBox.left * imageWidth,
                detection.boundingBox.top * imageHeight,
                detection.boundingBox.right * imageWidth,
                detection.boundingBox.bottom * imageHeight
            )
            
            detection.copy(boundingBox = scaledBox)
        }
    }
    
    /**
     * Load metadata.yaml file to get class names and model info
     */
    private fun loadMetadata(metadataPath: String) {
        try {
            Log.d(TAG, "Opening metadata file: $metadataPath")
            val yaml = Yaml()
            val inputStream = context.assets.open(metadataPath)
            
            Log.d(TAG, "Parsing YAML content...")
            val data = yaml.load<Map<String, Any>>(inputStream)
            
            // Extract class names
            val names = data["names"] as? Map<Int, String>
            if (names != null) {
                classNames = Array(names.size) { i -> names[i] ?: "Unknown" }
                Log.d(TAG, "Loaded ${classNames.size} class names")
            } else {
                Log.w(TAG, "No class names found in metadata, using default")
                classNames = arrayOf("person", "bicycle", "car", "motorcycle", "airplane") // Default COCO classes subset
            }
            
            // Extract image size if available
            val imgSz = data["imgsz"] as? List<Int>
            if (imgSz != null && imgSz.size >= 2) {
                inputSize = maxOf(imgSz[0], imgSz[1])
                Log.d(TAG, "Set input size from metadata: $inputSize")
            } else {
                Log.d(TAG, "Using default input size: $inputSize")
            }
            
            inputStream.close()
            
            Log.d(TAG, "Loaded metadata: ${classNames.size} classes, input size: $inputSize")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading metadata: ${e.message}", e)
            // Use default values
            classNames = arrayOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light")
            inputSize = 640
            Log.d(TAG, "Using default metadata: ${classNames.size} classes, input size: $inputSize")
        }
    }
    
    /**
     * Load model file from assets
     */
    private fun loadModelFile(modelPath: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        synchronized(detectionLock) {
            // Wait for any ongoing detection to complete
            while (isDetecting) {
                try {
                    Thread.sleep(10)
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    break
                }
            }
            
            interpreter?.close()
            interpreter = null
        }
    }
}