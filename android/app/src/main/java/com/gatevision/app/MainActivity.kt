@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import coil.ImageLoader

class MainActivity : ComponentActivity() {
    private val vm: GateVm by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GateVisionTheme {
                GateVisionApp(vm)
            }
        }
    }

    // Poll the server only while the app is on screen (saves battery and mobile data).
    override fun onResume() {
        super.onResume()
        vm.startLive()
    }

    override fun onPause() {
        vm.stopLive()
        super.onPause()
    }
}

@Composable
fun GateVisionApp(vm: GateVm) {
    val session = vm.session
    if (session == null) {
        SetupScreen(vm)
        return
    }

    val context = LocalContext.current
    val loader = remember(session.token) {
        ImageLoader.Builder(context)
            .okHttpClient {
                ApiClient.http.newBuilder()
                    .addInterceptor { chain ->
                        chain.proceed(
                            chain.request().newBuilder()
                                .header("Authorization", "Bearer ${session.token}")
                                .build(),
                        )
                    }
                    .build()
            }
            .respectCacheHeaders(false)
            .crossfade(true)
            .build()
    }

    // a guard account only sees the live list; managers and admins get everything
    val full = session.role == "admin" || session.role == "manager"
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var detail by remember { mutableStateOf<Event?>(null) }
    var showSettings by remember { mutableStateOf(false) }

    // make sure polling runs once the user has signed in (onResume already ran before that)
    androidx.compose.runtime.LaunchedEffect(session.token) { vm.startLive() }

    CompositionLocalProvider(LocalImageLoader provides loader) {
        val open = detail
        if (open != null) {
            BackHandler { detail = null }
            DetailScreen(
                vm = vm,
                e = open,
                canSearch = full,
                onBack = { detail = null },
                onFindPlate = { plate ->
                    detail = null
                    tab = 1
                    vm.searchPlate(plate)
                },
            )
        } else {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("Gate Vision") },
                        actions = {
                            IconButton(onClick = { showSettings = true }) {
                                Icon(Icons.Default.Settings, contentDescription = "Settings")
                            }
                        },
                    )
                },
                bottomBar = {
                    if (full) {
                        NavigationBar {
                            NavigationBarItem(
                                selected = tab == 0, onClick = { tab = 0 },
                                icon = { Icon(Icons.Default.PlayArrow, contentDescription = null) },
                                label = { Text("Live") },
                            )
                            NavigationBarItem(
                                selected = tab == 1, onClick = { tab = 1 },
                                icon = { Icon(Icons.Default.Search, contentDescription = null) },
                                label = { Text("Search") },
                            )
                            NavigationBarItem(
                                selected = tab == 2, onClick = { tab = 2 },
                                icon = { Icon(Icons.Default.Info, contentDescription = null) },
                                label = { Text("Cameras") },
                            )
                        }
                    }
                },
            ) { padding ->
                androidx.compose.foundation.layout.Box(Modifier.padding(padding)) {
                    when (if (full) tab else 0) {
                        0 -> LiveScreen(vm) { detail = it }
                        1 -> SearchScreen(vm) { detail = it }
                        else -> CamerasScreen(vm)
                    }
                }
            }
        }

        if (showSettings) {
            AlertDialog(
                onDismissRequest = { showSettings = false },
                title = { Text("Settings") },
                text = {
                    Text(
                        "Server: ${session.baseUrl}\nSigned in as: ${session.username} (${session.role})\n" +
                            "App version: ${appVersion(context)}",
                    )
                },
                confirmButton = {
                    TextButton(onClick = { showSettings = false; vm.logout() }) { Text("Sign out") }
                },
                dismissButton = { TextButton(onClick = { showSettings = false }) { Text("Close") } },
            )
        }
    }
}

private fun appVersion(context: android.content.Context): String =
    try {
        context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "?"
    } catch (e: Exception) {
        "?"
    }
