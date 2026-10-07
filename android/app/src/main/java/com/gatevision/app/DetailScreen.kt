@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.horizontalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage

@Composable
fun DetailScreen(vm: GateVm, e: Event, canSearch: Boolean, onBack: () -> Unit, onFindPlate: (String) -> Unit) {
    val pictures = listOfNotNull(
        e.fullPath?.let { "Scene" to it },
        e.cropPath?.let { (if (e.isVehicle) "Vehicle" else "Person") to it },
        e.platePath?.let { "Plate" to it },
        e.facePath?.let { "Face" to it },
    )
    // start on the most informative picture
    val firstKey = if (e.isVehicle) "Plate" else "Face"
    var selected by remember { mutableStateOf(pictures.firstOrNull { it.first == firstKey }?.first ?: pictures.firstOrNull()?.first) }
    var zoom by remember { mutableStateOf(false) }
    val currentPath = pictures.firstOrNull { it.first == selected }?.second
    val currentUrl = vm.mediaUrl(currentPath)

    if (zoom && currentUrl != null) {
        ZoomDialog(currentUrl) { zoom = false }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(e.title(), fontFamily = if (e.isVehicle) FontFamily.Monospace else FontFamily.Default) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(14.dp),
        ) {
            if (currentUrl != null) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(14.dp))
                        .background(Color.Black)
                        .clickable { zoom = true },
                    contentAlignment = Alignment.Center,
                ) {
                    AsyncImage(
                        model = currentUrl,
                        contentDescription = null,
                        imageLoader = LocalImageLoader.current,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 160.dp, max = 360.dp),
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
                    pictures.forEach { (label, _) ->
                        FilterChip(selected = selected == label, onClick = { selected = label }, label = { Text(label) })
                    }
                }
                Text("Tap the picture to enlarge and zoom.", fontSize = 12.sp, color = MaterialTheme.colorScheme.secondary)
            } else {
                EmptyState("No pictures saved for this record.")
            }

            Spacer(Modifier.height(16.dp))
            InfoRow("Time", formatTime(e.ts))
            InfoRow("Direction", e.direction)
            InfoRow("Camera", e.cameraName)
            if (e.isVehicle) {
                InfoRow("Plate", e.plateText ?: "not read")
                e.plateConf?.let { InfoRow("Plate confidence", "${(it * 100).toInt()}%") }
                InfoRow("Vehicle", e.vehicleType ?: "-")
                InfoRow("Colour", e.color ?: "-")
            } else {
                InfoRow("Top colour", e.upperColor ?: "-")
                InfoRow("Bottom colour", e.lowerColor ?: "-")
            }
            val plate = e.plateText
            if (canSearch && e.isVehicle && plate != null) {
                Spacer(Modifier.height(16.dp))
                Button(onClick = { onFindPlate(plate) }, modifier = Modifier.fillMaxWidth().height(48.dp)) {
                    Text("Find every record of this plate (30 days)")
                }
            }
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Text(label, color = MaterialTheme.colorScheme.secondary, modifier = Modifier.width(130.dp))
        Text(value, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ZoomDialog(url: String, onClose: () -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(Modifier.fillMaxSize().background(Color.Black)) {
            AsyncImage(
                model = url,
                contentDescription = null,
                imageLoader = LocalImageLoader.current,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, zoomChange, _ ->
                            scale = (scale * zoomChange).coerceIn(1f, 6f)
                            if (scale == 1f) {
                                offsetX = 0f
                                offsetY = 0f
                            } else {
                                offsetX += pan.x
                                offsetY += pan.y
                            }
                        }
                    }
                    .graphicsLayer(
                        scaleX = scale,
                        scaleY = scale,
                        translationX = offsetX,
                        translationY = offsetY,
                    ),
            )
            IconButton(onClick = onClose, modifier = Modifier.align(Alignment.TopEnd).padding(12.dp)) {
                Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
            }
        }
    }
}
