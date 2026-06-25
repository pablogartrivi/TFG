package com.example.polarmetria

import androidx.activity.ComponentActivity
import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.*
import androidx.compose.ui.unit.dp
import android.view.TextureView
import android.view.Surface
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.polarmetria.ui.theme.PolarímetriaTheme
import android.hardware.camera2.*
import android.media.ImageReader
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.DngCreator
import android.provider.MediaStore
import androidx.annotation.RequiresPermission
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.foundation.shape.CircleShape
import kotlinx.coroutines.delay

data class CameraSettings(
    var iso: Int = 800,
    var exposureTimeNs: Long = 10_000_000L, // default 1/100s
    var autoFocus: Boolean = true,
    var mode: CaptureMode = CaptureMode.SINGLE,
    var zoomRatio: Float = 1.0f,       // Control de ampliación para enfoque
    var focusDistance: Float = 0.0f    // 0.0f = Infinito, valores mayores = más cerca
)

enum class PSG {
    V, H, P45, M45, RHP, LHP
}

enum class PSA(val label: String) {
    M1("M1"), M2("M2"), M3("M3"),
    M4("M4"), M5("M5"), M6("M6")
}

enum class CaptureMode {
    SINGLE,
    EXPOSURE_SWEEP,
    DARK_FRAMES,
    INTENSITY_CAPTURE,
    FIXED_BURST_30,
    BIOLOGICAL_CAPTURE
}

val exposureTimesSeconds = listOf(
    1.0, 0.7, 0.5, 0.3, 0.2, 0.1,
    1.0/20.0, 1.0/25.0, 1.0/30.0, 1.0/40.0, 1.0/45.0, 1.0/50.0,
    1.0/60.0, 1.0/70.0, 1.0/80.0, 1.0/90.0, 1.0/100.0, 1.0/200.0,
    1.0/300.0, 1.0/400.0, 1.0/500.0, 1.0/600.0, 1.0/700.0, 1.0/800.0,
    1.0/900.0, 1.0/1000.0, 1.0/2000.0, 1.0/3000.0, 1.0/4000.0, 1.0/5000.0,
    1.0/6000.0, 1.0/7000.0, 1.0/8000.0, 1.0/9000.0, 1.0/10000.0,
)

fun secondsToNs(seconds: Double): Long {
    return (seconds * 1_000_000_000L).toLong()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            PolarímetriaTheme {
                RequestCameraPermission {
                    var screen by remember { mutableStateOf("camera") }
                    val cameraSettings = remember { mutableStateOf(CameraSettings()) }

                    when (screen) {
                        "camera" -> CameraScreen(
                            onOpenSettings = { screen = "settings" },
                            settings = cameraSettings.value,
                            onUpdate = { cameraSettings.value = it }
                        )
                        "settings" -> SettingsScreen(
                            onBack = { screen = "camera" },
                            settings = cameraSettings.value,
                            onUpdate = { cameraSettings.value = it }
                        )
                    }
                }
            }
        }
    }
}

data class PolarState(
    val psg: PSG,
    val psa: PSA
)

fun generatePolarSequence(): List<PolarState> {
    val list = mutableListOf<PolarState>()
    PSG.values().forEach { psg ->
        PSA.values().forEach { psa ->
            list.add(PolarState(psg, psa))
        }
    }
    return list
}

@Composable
fun RequestCameraPermission(content: @Composable () -> Unit) {
    val context = LocalContext.current
    val activity = context as ComponentActivity

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    if (hasPermission) {
        content()
    } else {
        LaunchedEffect(Unit) {
            activity.requestPermissions(arrayOf(Manifest.permission.CAMERA), 0)
        }
    }
}

fun formatExposure(expNs: Long): String {
    return if (expNs >= 1_000_000_000L) {
        "${expNs / 1_000_000_000}s"
    } else {
        "1/${1_000_000_000L / expNs}s"
    }
}

fun takeFixedBurst30(
    context: Context,
    cameraDevice: CameraDevice,
    captureSession: CameraCaptureSession,
    imageReader: ImageReader,
    characteristics: CameraCharacteristics,
    settings: CameraSettings,
    onProgress: (Int) -> Unit,
    onComplete: () -> Unit
) {
    val total = 30
    var index = 0

    fun captureNext() {
        if (index >= total) {
            onComplete()
            return
        }

        val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            set(CaptureRequest.SENSOR_SENSITIVITY, settings.iso)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs)
            set(CaptureRequest.SENSOR_FRAME_DURATION, settings.exposureTimeNs)
            set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
            set(CaptureRequest.SHADING_MODE, CaptureRequest.SHADING_MODE_OFF)
            set(CaptureRequest.HOT_PIXEL_MODE, CaptureRequest.HOT_PIXEL_MODE_OFF)
            set(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_OFF)
            addTarget(imageReader.surface)
        }

        captureSession.capture(
            request.build(),
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    val image = imageReader.acquireNextImage()
                    if (image != null) {
                        val expMs = settings.exposureTimeNs / 1_000_000
                        val name = "EXP${expMs}ms_ISO${settings.iso}_${String.format("%03d", index)}.dng"
                        val values = ContentValues().apply {
                            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                            put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng")
                            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Polarimetria")
                        }
                        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                        uri?.let {
                            val dngCreator = DngCreator(characteristics, result)
                            context.contentResolver.openOutputStream(it)?.use { output ->
                                dngCreator.writeImage(output, image)
                            }
                        }
                        image.close()
                    }
                    onProgress(index)
                    index++
                    captureNext()
                }
            },
            null
        )
    }
    captureNext()
}

@Composable
fun CameraScreen(
    onOpenSettings: () -> Unit,
    settings: CameraSettings,
    onUpdate: (CameraSettings) -> Unit
) {
    var isSweeping by remember { mutableStateOf(false) }
    var sweepProgress by remember { mutableStateOf(0) }
    var sweepTotal by remember { mutableStateOf(exposureTimesSeconds.size) }
    var sweepFinished by remember { mutableStateOf(false) }

    var captureSession by remember { mutableStateOf<CameraCaptureSession?>(null) }
    val context = LocalContext.current
    val textureView = remember { TextureView(context) }
    val cameraManager = remember { context.getSystemService(Context.CAMERA_SERVICE) as CameraManager }

    var cameraDevice by remember { mutableStateOf<CameraDevice?>(null) }
    var isCameraReady by remember { mutableStateOf(false) }
    var previewSurface by remember { mutableStateOf<Surface?>(null) }
    val cameraId = remember { cameraManager.cameraIdList[0] }
    val characteristics = remember { cameraManager.getCameraCharacteristics(cameraId) }
    val polarSequence = remember { generatePolarSequence() }
    var polarIndex by remember { mutableStateOf(0) }
    val maxZoom = characteristics.get(
        CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM
    ) ?: 1.0f
    cameraManager.cameraIdList.forEach { id ->
        val chars = cameraManager.getCameraCharacteristics(id)

        val lensFacing = chars.get(CameraCharacteristics.LENS_FACING)
        val maxZoom = chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)

        Log.d("CAMERA_ZOOM", "ID=$id LENS=$lensFacing MAX_ZOOM=$maxZoom")
    }
    CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE

    val minFocus = characteristics.get(
        CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE
    ) ?: 0f

    Log.d("FOCUS", "MIN_FOCUS_DISTANCE = $minFocus")

    var bioImageCount by remember { mutableStateOf(1) }
    var showCaptureConfirmation by remember { mutableStateOf(false) }

    var histogramData by remember { mutableStateOf(IntArray(256)) }

    val imageReader = remember {
        val size = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            .getOutputSizes(ImageFormat.RAW_SENSOR)[0]
        ImageReader.newInstance(size.width, size.height, ImageFormat.RAW_SENSOR, 2)
    }

    val histogramReader = remember {
        ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2)
    }

    LaunchedEffect(showCaptureConfirmation) {
        if (showCaptureConfirmation) {
            delay(2000)
            showCaptureConfirmation = false
        }
    }

    LaunchedEffect(histogramReader) {
        histogramReader.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                val buffer = image.planes[0].buffer
                val rowStride = image.planes[0].rowStride
                val pixelStride = image.planes[0].pixelStride
                val w = image.width
                val h = image.height
                val bins = IntArray(256)

                for (y in 0 until h step 6) {
                    for (x in 0 until w step 6) {
                        val offset = y * rowStride + x * pixelStride
                        if (offset < buffer.limit()) {
                            val luma = buffer.get(offset).toInt() and 0xFF
                            bins[luma]++
                        }
                    }
                }
                histogramData = bins
            } catch (e: Exception) {
                Log.e("Histogram", "Error en cálculo", e)
            } finally {
                image.close()
            }
        }, null)
    }

    LaunchedEffect(captureSession, previewSurface, settings.iso, settings.exposureTimeNs, settings.zoomRatio, settings.focusDistance, settings.mode) {
        val session = captureSession ?: return@LaunchedEffect
        val surface = previewSurface ?: return@LaunchedEffect
        val device = cameraDevice ?: return@LaunchedEffect

        try {
            val requestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                if (settings.mode == CaptureMode.BIOLOGICAL_CAPTURE) {
                    addTarget(histogramReader.surface)
                }

                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, settings.iso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs)
                set(CaptureRequest.SENSOR_FRAME_DURATION, settings.exposureTimeNs)

                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                set(CaptureRequest.LENS_FOCUS_DISTANCE, settings.focusDistance)

                val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                if (sensorSize != null) {
                    val xCenter = sensorSize.width() / 2
                    val yCenter = sensorSize.height() / 2
                    val xDelta = (sensorSize.width() / (2 * settings.zoomRatio)).toInt()
                    val yDelta = (sensorSize.height() / (2 * settings.zoomRatio)).toInt()
                    set(CaptureRequest.SCALER_CROP_REGION, Rect(xCenter - xDelta, yCenter - yDelta, xCenter + xDelta, yCenter + yDelta))
                }

                set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
                set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
            }
            session.setRepeatingRequest(requestBuilder.build(), null, null)
        } catch (e: Exception) {
            Log.e("Camera2", "Error actualizando parámetros repetitivos", e)
        }
    }

    LaunchedEffect(Unit) {
        val permissionGranted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (!permissionGranted) return@LaunchedEffect

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                val surfaceTexture = textureView.surfaceTexture!!
                surfaceTexture.setDefaultBufferSize(1920, 1080)
                val surface = Surface(surfaceTexture)
                previewSurface = surface

                camera.createCaptureSession(
                    listOf(surface, imageReader.surface, histogramReader.surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            captureSession = session
                            isCameraReady = true
                        }
                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            Log.e("Camera2", "Config Failed")
                        }
                    }, null
                )
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close() }
            override fun onError(camera: CameraDevice, error: Int) { camera.close() }
        }, null)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { textureView }, modifier = Modifier.fillMaxSize())

        Surface(
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 140.dp, end = 16.dp),
            tonalElevation = 8.dp,
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.6f)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(text = "ISO: ${settings.iso}", color = androidx.compose.ui.graphics.Color.White)
                Text(text = "EXP: ${formatExposure(settings.exposureTimeNs)}", color = androidx.compose.ui.graphics.Color.White)
                if(settings.mode == CaptureMode.BIOLOGICAL_CAPTURE) {
                    Text(text = "MODO: BIO (RAW ÚNICO)", color = androidx.compose.ui.graphics.Color.Green, style = MaterialTheme.typography.labelSmall)
                    Text(text = "Siguiente Nº: $bioImageCount", color = androidx.compose.ui.graphics.Color.Cyan, style = MaterialTheme.typography.labelSmall)
                }
            }
        }

        Surface(
            onClick = onOpenSettings,
            modifier = Modifier.align(Alignment.TopEnd).padding(16.dp).size(56.dp),
            shape = MaterialTheme.shapes.medium,
            tonalElevation = 6.dp,
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(imageVector = Icons.Default.Settings, contentDescription = "Settings", modifier = Modifier.size(30.dp))
            }
        }

        if (settings.mode == CaptureMode.BIOLOGICAL_CAPTURE) {
            Surface(
                modifier = Modifier.align(Alignment.CenterStart).padding(start = 16.dp).width(140.dp),
                tonalElevation = 6.dp,
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.75f)
            ) {
                val min = 1f
                val max = 30f

                fun toSliderValue(real: Float): Float {
                    val t = (real - min) / (max - min)
                    return t * t
                }

                fun fromSliderValue(slider: Float): Float {
                    return min + (max - min) * kotlin.math.sqrt(slider)
                }
                fun toSliderFocus(real: Float, max: Float): Float {
                    val t = real / max
                    return kotlin.math.sqrt(t)
                }

                fun fromSliderFocus(slider: Float, max: Float): Float {
                    return (slider * slider) * max
                }
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("🔎 Ampliar: ${String.format("%.1fx", settings.zoomRatio)}", style = MaterialTheme.typography.bodySmall)
                    Slider(
                        value = toSliderValue(settings.zoomRatio),
                        onValueChange = {
                            onUpdate(settings.copy(
                                zoomRatio = fromSliderValue(it)
                            ))
                        },
                        valueRange = 0f..1f
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f))
                    Text(" Foco M: ${String.format("%.2f", settings.focusDistance)}", style = MaterialTheme.typography.bodySmall)
                    Slider(
                        value = toSliderFocus(settings.focusDistance, 9.5f),
                        onValueChange = {
                            onUpdate(settings.copy(
                                focusDistance = fromSliderFocus(it, 9.5f)
                            ))
                        },
                        valueRange = 0f..1f
                    )
                }
            }

            HistogramOverlay(
                histogramData = histogramData,
                modifier = Modifier.align(Alignment.TopStart).padding(top = 40.dp, start = 16.dp)
            )
        }

        if (settings.mode == CaptureMode.INTENSITY_CAPTURE) {
            val currentPolar = polarSequence.getOrNull(polarIndex)
            currentPolar?.let { state ->
                Surface(
                    modifier = Modifier.align(Alignment.TopStart).padding(top = 40.dp, start = 16.dp),
                    tonalElevation = 8.dp,
                    color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                    contentColor = MaterialTheme.colorScheme.onSurface,
                    shape = MaterialTheme.shapes.medium,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary)
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = " GUÍA DE INTENSIDADES",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(text = "Ajustar PSG ➡️  ${state.psg}", style = MaterialTheme.typography.bodyMedium)
                        Text(text = "Ajustar PSA ➡️  ${state.psa.label}", style = MaterialTheme.typography.bodyMedium)
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Estado: ${polarIndex + 1} / ${polarSequence.size}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                }
            }
        }

        // Indicadores de otros modos
        if (isSweeping) {
            Surface(modifier = Modifier.align(Alignment.TopCenter).padding(top = 40.dp), tonalElevation = 8.dp) {
                Column(modifier = Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Barrido en curso"); Text("$sweepProgress / $sweepTotal")
                }
            }
        }
        if (sweepFinished) {
            Surface(modifier = Modifier.align(Alignment.TopCenter).padding(top = 40.dp), tonalElevation = 8.dp) {
                Text("Barrido completado", modifier = Modifier.padding(12.dp))
            }
        }

        if (showCaptureConfirmation) {
            Surface(
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 120.dp),
                color = MaterialTheme.colorScheme.primaryContainer,
                shape = MaterialTheme.shapes.large,
                tonalElevation = 8.dp,
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "¡Captura Guardada Exitosamente! (Nº ${bioImageCount - 1})",
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }

        Button(
            onClick = {
                cameraDevice?.let { device ->
                    captureSession?.let { session ->
                        imageReader?.let { reader ->
                            when (settings.mode) {
                                CaptureMode.SINGLE, CaptureMode.BIOLOGICAL_CAPTURE -> {
                                    takePhotoRAW(context, settings, device, session, reader, characteristics, bioImageCount) {
                                        bioImageCount++
                                        showCaptureConfirmation = true
                                    }
                                }
                                CaptureMode.INTENSITY_CAPTURE -> {
                                    val current = polarSequence.getOrNull(polarIndex)
                                    current?.let { state ->
                                        takeSingleIntensityPhoto(context, device, session, reader, characteristics, settings, state, onCaptured = {
                                            polarIndex = (polarIndex + 1) % polarSequence.size
                                        })
                                    }
                                }
                                CaptureMode.EXPOSURE_SWEEP -> {
                                    takeExposureSweep(context, device, session, reader, characteristics, false,
                                        onStart = { isSweeping = true; sweepFinished = false; sweepProgress = 0 },
                                        onProgress = { i -> sweepProgress = i },
                                        onComplete = { isSweeping = false; sweepFinished = true }
                                    )
                                }
                                CaptureMode.DARK_FRAMES -> {
                                    takeExposureSweep(context, device, session, reader, characteristics, true,
                                        onStart = { isSweeping = true; sweepFinished = false; sweepProgress = 0 },
                                        onProgress = { i -> sweepProgress = i },
                                        onComplete = { isSweeping = false; sweepFinished = true }
                                    )
                                }
                                CaptureMode.FIXED_BURST_30 -> {
                                    takeFixedBurst30(context, device, session, reader, characteristics, settings,
                                        onProgress = { i -> Log.d("BURST", "Foto $i / 30") },
                                        onComplete = { Log.d("BURST", "Ráfaga completada") }
                                    )
                                }
                            }
                        }
                    }
                }
            },
            enabled = isCameraReady,
            modifier = Modifier.align(Alignment.Center).size(180.dp),
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.4f))
        ) {
            Text("●", style = MaterialTheme.typography.headlineLarge)
        }
    }
}

@Composable
fun HistogramOverlay(histogramData: IntArray, modifier: Modifier = Modifier) {
    val maxBinValue = remember(histogramData) { (histogramData.maxOrNull() ?: 1).coerceAtLeast(1) }
    // Alerta si más del 0.8% de los píxeles muestreados están totalmente saturados en el límite 255
    val isSaturating = remember(histogramData) { histogramData[255] > (histogramData.sum() * 0.008) }

    Surface(
        modifier = modifier.size(width = 170.dp, height = 110.dp),
        color = androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.6f),
        shape = MaterialTheme.shapes.small,
        border = BorderStroke(1.dp, if (isSaturating) androidx.compose.ui.graphics.Color.Red else androidx.compose.ui.graphics.Color.LightGray)
    ) {
        Column(modifier = Modifier.padding(6.dp)) {
            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    val barW = w / 256f

                    for (i in 0 until 256) {
                        val barH = (histogramData[i].toFloat() / maxBinValue) * h
                        drawRect(
                            color = if (i >= 250) androidx.compose.ui.graphics.Color.Red else androidx.compose.ui.graphics.Color.White,
                            topLeft = androidx.compose.ui.geometry.Offset(x = i * barW, y = h - barH),
                            size = androidx.compose.ui.geometry.Size(width = barW.coerceAtLeast(1f), height = barH)
                        )
                    }
                }
                if (isSaturating) {
                    Text(
                        text = "⚠️ SATURADO",
                        color = androidx.compose.ui.graphics.Color.Red,
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.align(Alignment.TopEnd).background(androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.7f)).padding(2.dp)
                    )
                }
            }
            Text(
                text = "Exposición RAW (Luma)",
                color = androidx.compose.ui.graphics.Color.White,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 2.dp)
            )
        }
    }
}

@RequiresPermission(Manifest.permission.CAMERA)
fun takePhotoRAW(
    context: Context,
    settings: CameraSettings,
    cameraDevice: CameraDevice,
    captureSession: CameraCaptureSession,
    imageReader: ImageReader,
    characteristics: CameraCharacteristics,
    currentCount: Int,
    onCaptureSaved: () -> Unit
) {
    var pendingResult: TotalCaptureResult? = null
    var pendingImage: android.media.Image? = null

    fun tryProcess() {
        val image = pendingImage
        val result = pendingResult
        if (image == null || result == null) return

        val prefix = if (settings.mode == CaptureMode.BIOLOGICAL_CAPTURE) "BIO_MANUAL" else "IMG"
        val formattedCounter = String.format("%03d", currentCount)

        val name = "${prefix}_${formattedCounter}_ISO${settings.iso}_EXP${settings.exposureTimeNs}_Z${String.format("%.1f", settings.zoomRatio)}_F${String.format("%.1f", settings.focusDistance)}_${System.currentTimeMillis()}.dng"

        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Polarimetria")
        }

        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) { image.close(); return }

        try {
            val dngCreator = DngCreator(characteristics, result)
            context.contentResolver.openOutputStream(uri)?.use { output ->
                dngCreator.writeImage(output, image)
            }
            Log.d("Camera2", "Guardado con éxito: $name")

            onCaptureSaved()
        } catch (e: Exception) {
            Log.e("Camera2", "Error DNG", e)
        } finally {
            image.close()
            pendingImage = null
            pendingResult = null
        }
    }

    imageReader.setOnImageAvailableListener({ reader ->
        pendingImage = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
        tryProcess()
    }, null)

    val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
        set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF)
        set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)

        // Mantener enfoque manual estricto en la captura física
        set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
        set(CaptureRequest.LENS_FOCUS_DISTANCE, settings.focusDistance)

        val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        if (sensorSize != null) {
            val xCenter = sensorSize.width() / 2
            val yCenter = sensorSize.height() / 2
            val xDelta = (sensorSize.width() / (2 * settings.zoomRatio)).toInt()
            val yDelta = (sensorSize.height() / (2 * settings.zoomRatio)).toInt()
            set(CaptureRequest.SCALER_CROP_REGION, Rect(xCenter - xDelta, yCenter - yDelta, xCenter + xDelta, yCenter + yDelta))
        }

        set(CaptureRequest.SENSOR_SENSITIVITY, settings.iso)
        set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs)
        set(CaptureRequest.SENSOR_FRAME_DURATION, settings.exposureTimeNs)

        set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
        set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
        set(CaptureRequest.SHADING_MODE, CaptureRequest.SHADING_MODE_OFF)
        set(CaptureRequest.HOT_PIXEL_MODE, CaptureRequest.HOT_PIXEL_MODE_OFF)
        set(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_OFF)

        addTarget(imageReader.surface)
    }

    captureSession.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            pendingResult = result
            tryProcess()
        }
    }, null)
}

fun takeExposureSweep(context: Context, cameraDevice: CameraDevice, captureSession: CameraCaptureSession, imageReader: ImageReader, characteristics: CameraCharacteristics, isDark: Boolean, onStart: () -> Unit, onProgress: (Int) -> Unit, onComplete: () -> Unit) {
    onStart(); val isoFixed = 21; val exposureListNs = exposureTimesSeconds.map { secondsToNs(it) }; var index = 0
    fun captureNext() {
        if (index >= exposureListNs.size) { onComplete(); return }
        val exposure = exposureListNs[index]
        val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            set(CaptureRequest.SENSOR_SENSITIVITY, isoFixed)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposure)
            set(CaptureRequest.SENSOR_FRAME_DURATION, exposure)
            set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
            set(CaptureRequest.SHADING_MODE, CaptureRequest.SHADING_MODE_OFF)
            addTarget(imageReader.surface)
        }
        captureSession.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val image = imageReader.acquireLatestImage()
                if (image != null) {
                    val suffix = if (isDark) "_d" else ""
                    val name = "EXP_${exposure}_ns_ISO${isoFixed}${suffix}.dng"
                    val values = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                        put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng"); put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Polarimetria")
                    }
                    val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    uri?.let {
                        val dngCreator = DngCreator(characteristics, result)
                        context.contentResolver.openOutputStream(it)?.use { out -> dngCreator.writeImage(out, image) }
                    }
                    image.close()
                }
                onProgress(index); index++; captureNext()
            }
        }, null)
    }
    captureNext()
}

fun takeSingleIntensityPhoto(context: Context, cameraDevice: CameraDevice, captureSession: CameraCaptureSession, imageReader: ImageReader, characteristics: CameraCharacteristics, settings: CameraSettings, polarState: PolarState, onCaptured: () -> Unit) {
    val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
        set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF); set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF); set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF); set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
        set(CaptureRequest.SENSOR_SENSITIVITY, settings.iso); set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs); set(CaptureRequest.SENSOR_FRAME_DURATION, settings.exposureTimeNs)
        set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF); set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
        addTarget(imageReader.surface)
    }
    captureSession.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val image = imageReader.acquireLatestImage()
            if (image != null) {
                val name = "PSG_${polarState.psg}_PSA_${polarState.psa.label}_ISO${settings.iso}_EXP${settings.exposureTimeNs}.dng"
                val values = ContentValues().apply { put(MediaStore.MediaColumns.DISPLAY_NAME, name); put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng"); put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Polarimetria") }
                val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                uri?.let { val dc = DngCreator(characteristics, result); context.contentResolver.openOutputStream(it)?.use { out -> dc.writeImage(out, image) } }
                image.close()
            }
            onCaptured()
        }
    }, null)
}

fun takeIntensitySeries(context: Context, cameraDevice: CameraDevice, captureSession: CameraCaptureSession, imageReader: ImageReader, characteristics: CameraCharacteristics, settings: CameraSettings, polarSequence: List<PolarState>, onStateChange: (Int) -> Unit, numImages: Int = 10, onStart: () -> Unit, onProgress: (Int) -> Unit, onComplete: () -> Unit) {
    onStart(); val totalImg = polarSequence.size; var index = 0
    fun captureNext() {
        if (index >= totalImg) { onComplete(); return }
        val currentState = polarSequence[index]; onStateChange(index)
        val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF); set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF); set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            set(CaptureRequest.SENSOR_SENSITIVITY, settings.iso); set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs); set(CaptureRequest.SENSOR_FRAME_DURATION, settings.exposureTimeNs)
            addTarget(imageReader.surface)
        }
        captureSession.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val image = imageReader.acquireNextImage()
                if (image != null) {
                    val name = "PSG_${currentState.psg}_PSA_${currentState.psa.label}_ISO${settings.iso}_EXP${settings.exposureTimeNs}.dng"
                    val values = ContentValues().apply { put(MediaStore.MediaColumns.DISPLAY_NAME, name); put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng"); put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Polarimetria") }
                    val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    uri?.let { val dc = DngCreator(characteristics, result); context.contentResolver.openOutputStream(it)?.use { out -> dc.writeImage(out, image) } }
                    image.close()
                }
                onProgress(index); index++; captureNext()
            }
        }, null)
    }
    captureNext()
}

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    settings: CameraSettings,
    onUpdate: (CameraSettings) -> Unit
) {
    val scrollState = rememberScrollState()

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(top = 40.dp)
                .verticalScroll(scrollState)
        ) {
            Button(onClick = onBack, modifier = Modifier.padding(top = 24.dp, bottom = 20.dp)) { Text("Volver") }

            val minIso = 21; val maxIso = 3200
            Text("ISO seleccionado: ${settings.iso}", style = MaterialTheme.typography.titleMedium)
            Slider(
                value = settings.iso.toFloat().coerceIn(minIso.toFloat(), maxIso.toFloat()),
                onValueChange = { onUpdate(settings.copy(iso = it.toInt().coerceIn(minIso, maxIso))) },
                valueRange = minIso.toFloat()..maxIso.toFloat()
            )
            Spacer(Modifier.height(16.dp))

            val selectedIndex = remember { mutableStateOf(10) }
            val exposureNsList = exposureTimesSeconds.map { secondsToNs(it) }
            Text("Exposición seleccionada", style = MaterialTheme.typography.titleMedium)
            Text(text = "1/${(1.0 / exposureTimesSeconds[selectedIndex.value]).toInt()} s")
            Slider(
                value = selectedIndex.value.toFloat(),
                onValueChange = { selectedIndex.value = it.toInt(); onUpdate(settings.copy(exposureTimeNs = exposureNsList[selectedIndex.value])) },
                valueRange = 0f..(exposureTimesSeconds.size - 1).toFloat(),
                steps = exposureTimesSeconds.size - 2
            )

            Spacer(Modifier.height(24.dp))

            Surface(modifier = Modifier.fillMaxWidth(), tonalElevation = 4.dp, shape = MaterialTheme.shapes.medium) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Modo de captura activo", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(bottom = 12.dp))

                    val getColors = @Composable { mode: CaptureMode ->
                        if (settings.mode == mode) {
                            ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary, contentColor = MaterialTheme.colorScheme.onPrimary)
                        } else {
                            ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.surfaceVariant, contentColor = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }

                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { onUpdate(settings.copy(mode = CaptureMode.SINGLE)) }, colors = getColors(CaptureMode.SINGLE), modifier = Modifier.weight(1f)) { Text("Normal") }
                        Button(onClick = { onUpdate(settings.copy(mode = CaptureMode.EXPOSURE_SWEEP)) }, colors = getColors(CaptureMode.EXPOSURE_SWEEP), modifier = Modifier.weight(1f)) { Text("Barrido Exp.") }
                    }
                    Spacer(Modifier.height(8.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { onUpdate(settings.copy(mode = CaptureMode.DARK_FRAMES)) }, colors = getColors(CaptureMode.DARK_FRAMES), modifier = Modifier.weight(1f)) { Text("Barrido Darks") }
                        Button(onClick = { onUpdate(settings.copy(mode = CaptureMode.INTENSITY_CAPTURE)) }, colors = getColors(CaptureMode.INTENSITY_CAPTURE), modifier = Modifier.weight(1f)) { Text("Intens. Man.") }
                    }
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = { onUpdate(settings.copy(mode = CaptureMode.FIXED_BURST_30)) }, colors = getColors(CaptureMode.FIXED_BURST_30), modifier = Modifier.fillMaxWidth()) { Text("Burst 30") }

                    Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(12.dp))

                    Button(
                        onClick = { onUpdate(settings.copy(mode = CaptureMode.BIOLOGICAL_CAPTURE)) },
                        colors = getColors(CaptureMode.BIOLOGICAL_CAPTURE),
                        modifier = Modifier.fillMaxWidth(),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
                    ) {
                        Text("Captura Biológica (Manual RAW + Histograma)")
                    }
                }
            }
            Spacer(Modifier.height(40.dp))
        }
    }
}