@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

@Composable
fun CamerasScreen(vm: GateVm) {
    LaunchedEffect(Unit) {
        while (true) {
            vm.refreshCams()
            delay(5000)
        }
    }
    Column(Modifier.fillMaxSize()) {
        val err = vm.camsError
        if (err != null) {
            Text(err, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(16.dp))
        }
        if (vm.cams.isEmpty() && err == null) {
            EmptyState("Loading cameras...")
        }
        LazyColumn(Modifier.fillMaxSize()) {
            items(vm.cams, key = { it.id }) { c ->
                Card(
                    Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                ) {
                    Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                        StatusDot(c.online)
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text(c.name, fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
                            val age = c.ageSeconds
                            val fps = c.fps
                            Text(
                                listOfNotNull(
                                    c.role,
                                    if (c.online) "online" else "OFFLINE",
                                    if (fps != null && c.online) "%.1f fps".format(fps) else null,
                                    if (age != null && !c.online) "last frame ${age}s ago" else null,
                                ).joinToString("  •  "),
                                fontSize = 13.sp,
                                color = MaterialTheme.colorScheme.secondary,
                            )
                        }
                        DirChip(c.direction)
                    }
                }
            }
        }
    }
}
