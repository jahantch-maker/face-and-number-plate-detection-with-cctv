@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.ImageLoader
import coil.compose.AsyncImage
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Image loader that sends the login token with every picture request. */
val LocalImageLoader = staticCompositionLocalOf<ImageLoader> { error("ImageLoader not provided") }

fun formatTime(ts: Double): String =
    SimpleDateFormat("dd MMM, HH:mm:ss", Locale.getDefault()).format(Date((ts * 1000).toLong()))

fun formatAgo(ts: Double, nowMs: Long): String {
    val s = ((nowMs - (ts * 1000).toLong()) / 1000).coerceAtLeast(0)
    return when {
        s < 5 -> "just now"
        s < 60 -> "${s}s ago"
        s < 3600 -> "${s / 60}m ago"
        s < 86400 -> "${s / 3600}h ${(s % 3600) / 60}m ago"
        else -> "${s / 86400}d ago"
    }
}

fun Event.title(): String {
    if (!isVehicle) return "Person"
    val p = plateText ?: return "No plate read"
    val unsure = plateConf != null && plateConf < 0.5
    return if (unsure) "$p ?" else p
}

fun Event.description(): String {
    return if (isVehicle) {
        listOfNotNull(color, vehicleType).joinToString(" ").ifBlank { "vehicle" }
    } else {
        val parts = listOfNotNull(upperColor?.let { "$it top" }, lowerColor?.let { "$it bottom" })
        parts.joinToString(", ").ifBlank { "person" }
    }
}

@Composable
fun Thumb(url: String?, modifier: Modifier = Modifier, scale: ContentScale = ContentScale.Crop) {
    if (url == null) {
        Box(modifier.background(MaterialTheme.colorScheme.surfaceVariant), contentAlignment = Alignment.Center) {
            Text("-", color = MaterialTheme.colorScheme.secondary)
        }
    } else {
        AsyncImage(
            model = url,
            contentDescription = null,
            imageLoader = LocalImageLoader.current,
            contentScale = scale,
            modifier = modifier,
        )
    }
}

@Composable
fun DirChip(direction: String) {
    val color = if (direction == "IN") InGreen else OutAmber
    Box(
        Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(color)
            .padding(horizontal = 8.dp, vertical = 2.dp),
    ) {
        Text(direction, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun EventCard(
    e: Event,
    thumbUrl: String?,
    nowMs: Long,
    highlight: Boolean,
    onClick: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (highlight) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surface,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
            Thumb(
                thumbUrl,
                Modifier
                    .size(width = 96.dp, height = 72.dp)
                    .clip(RoundedCornerShape(10.dp)),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    DirChip(e.direction)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        e.title(),
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = if (e.isVehicle) FontFamily.Monospace else FontFamily.Default,
                        fontStyle = if (e.isVehicle && e.plateText == null) FontStyle.Italic else FontStyle.Normal,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    e.description(),
                    fontSize = 14.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    "${e.cameraName}  •  ${formatAgo(e.ts, nowMs)}",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.secondary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
fun StatusDot(online: Boolean) {
    Box(
        Modifier
            .size(10.dp)
            .clip(CircleShape)
            .background(if (online) OnlineGreen else OfflineRed),
    )
}

@Composable
fun SectionLabel(text: String) {
    Text(
        text,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.secondary,
        modifier = Modifier.padding(top = 10.dp, bottom = 4.dp),
    )
}

@Composable
fun EmptyState(text: String) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(text, color = MaterialTheme.colorScheme.secondary)
    }
}
