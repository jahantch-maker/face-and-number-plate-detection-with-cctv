package com.gatevision.app

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val InGreen = Color(0xFF16A34A)
val OutAmber = Color(0xFFD97706)
val OnlineGreen = Color(0xFF22C55E)
val OfflineRed = Color(0xFFEF4444)

private val LightColors = lightColorScheme(
    primary = Color(0xFF0F766E),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCCEFEA),
    onPrimaryContainer = Color(0xFF00201C),
    secondary = Color(0xFF475569),
    background = Color(0xFFF6F8F8),
    surface = Color(0xFFFFFFFF),
    surfaceVariant = Color(0xFFE6ECEB),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF2DD4BF),
    onPrimary = Color(0xFF00201C),
    primaryContainer = Color(0xFF0B4F49),
    onPrimaryContainer = Color(0xFFCCEFEA),
    secondary = Color(0xFF94A3B8),
    background = Color(0xFF0D1413),
    surface = Color(0xFF131C1B),
    surfaceVariant = Color(0xFF1E2A29),
)

@Composable
fun GateVisionTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content,
    )
}
