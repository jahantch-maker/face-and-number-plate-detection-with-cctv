@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

@Composable
fun LiveScreen(vm: GateVm, onOpen: (Event) -> Unit) {
    var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1000)
            nowMs = System.currentTimeMillis()
        }
    }
    var filter by remember { mutableStateOf("all") }

    val shown = vm.live.filter {
        when (filter) {
            "vehicle" -> it.isVehicle
            "person" -> !it.isVehicle
            "IN" -> it.direction == "IN"
            "OUT" -> it.direction == "OUT"
            else -> true
        }
    }

    Column(Modifier.fillMaxSize()) {
        // connection line
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val err = vm.liveError
            val okAgo = if (vm.lastLiveOkMs == 0L) null else ((nowMs - vm.lastLiveOkMs) / 1000).coerceAtLeast(0)
            StatusDot(err == null && vm.lastLiveOkMs != 0L)
            Spacer(Modifier.width(8.dp))
            Text(
                when {
                    err != null -> "Offline - retrying...  ($err)"
                    okAgo == null -> "Connecting..."
                    else -> "Live  •  updated ${okAgo}s ago"
                },
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.secondary,
                maxLines = 2,
            )
        }

        Row(
            Modifier
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            listOf(
                "all" to "All",
                "vehicle" to "Vehicles",
                "person" to "People",
                "IN" to "IN",
                "OUT" to "OUT",
            ).forEach { (key, label) ->
                FilterChip(selected = filter == key, onClick = { filter = key }, label = { Text(label) })
            }
        }

        if (shown.isEmpty()) {
            EmptyState(if (vm.live.isEmpty()) "Waiting for detections..." else "Nothing matches this filter yet.")
        } else {
            LazyColumn(Modifier.fillMaxSize()) {
                items(shown, key = { it.id }) { e ->
                    EventCard(
                        e = e,
                        thumbUrl = vm.mediaUrl(e.thumbPath),
                        nowMs = nowMs,
                        highlight = e.id in vm.fresh,
                        onClick = { onOpen(e) },
                    )
                }
            }
        }
    }
}
