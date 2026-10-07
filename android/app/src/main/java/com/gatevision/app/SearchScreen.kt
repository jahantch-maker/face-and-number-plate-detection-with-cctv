@file:OptIn(ExperimentalMaterial3Api::class)

package com.gatevision.app

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

@Composable
fun ChoiceRow(
    options: List<Pair<String, String>>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    Row(
        Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        options.forEach { (value, label) ->
            FilterChip(selected = selected == value, onClick = { onSelect(value) }, label = { Text(label) })
        }
    }
}

@Composable
fun FilterDropdown(
    label: String,
    value: String,
    options: List<Pair<String, String>>,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var open by remember { mutableStateOf(false) }
    val shownValue = options.firstOrNull { it.first == value }?.second ?: "Any"
    Column(modifier) {
        OutlinedButton(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) {
            Text("$label: $shownValue", maxLines = 1)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text("Any") }, onClick = { onSelect(""); open = false })
            options.forEach { (v, l) ->
                DropdownMenuItem(text = { Text(l) }, onClick = { onSelect(v); open = false })
            }
        }
    }
}

@Composable
fun SearchScreen(vm: GateVm, onOpen: (Event) -> Unit) {
    var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1000)
            nowMs = System.currentTimeMillis()
        }
    }
    var showDate by remember { mutableStateOf(false) }
    val f = vm.form
    val meta = vm.meta
    val colorOptions = (meta?.colors ?: emptyList()).map { it to it.replaceFirstChar { c -> c.uppercase() } }
    val typeOptions = (meta?.vehicleTypes ?: emptyList()).map { it to it.replaceFirstChar { c -> c.uppercase() } }
    val cameraOptions = (meta?.cameras ?: emptyList()).map { it.id to it.name }

    if (showDate) {
        val state = rememberDatePickerState()
        DatePickerDialog(
            onDismissRequest = { showDate = false },
            confirmButton = {
                TextButton(onClick = {
                    val picked = state.selectedDateMillis
                    if (picked != null) vm.form = f.copy(range = "day", dayMillis = picked)
                    showDate = false
                }) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = { showDate = false }) { Text("Cancel") } },
        ) { DatePicker(state = state) }
    }

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Column(Modifier.padding(horizontal = 14.dp, vertical = 8.dp)) {
                OutlinedTextField(
                    value = f.plate,
                    onValueChange = { vm.form = f.copy(plate = it.uppercase()) },
                    label = { Text("Number plate (full or part)") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Characters,
                        imeAction = ImeAction.Search,
                    ),
                    keyboardActions = KeyboardActions(onSearch = { vm.runSearch() }),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = f.fuzzy, onCheckedChange = { vm.form = f.copy(fuzzy = it) })
                    Spacer(Modifier.width(10.dp))
                    Text("Similar plates (allows one wrong character)", fontSize = 13.sp)
                }

                SectionLabel("WHO")
                ChoiceRow(
                    listOf("" to "Anyone", "vehicle" to "Vehicles", "person" to "People"),
                    f.kind,
                ) { vm.form = f.copy(kind = it) }

                SectionLabel("DIRECTION")
                ChoiceRow(listOf("" to "Both", "IN" to "IN", "OUT" to "OUT"), f.direction) {
                    vm.form = f.copy(direction = it)
                }

                SectionLabel("WHEN")
                ChoiceRow(
                    listOf(
                        "1h" to "1 hour", "6h" to "6 hours", "24h" to "24 hours",
                        "7d" to "7 days", "30d" to "30 days", "all" to "All",
                    ),
                    f.range,
                ) { vm.form = f.copy(range = it) }
                Spacer(Modifier.height(6.dp))
                val dayLabel = f.dayMillis?.let {
                    Instant.ofEpochMilli(it).atZone(ZoneOffset.UTC).toLocalDate()
                        .format(DateTimeFormatter.ofPattern("dd MMM yyyy"))
                }
                OutlinedButton(onClick = { showDate = true }) {
                    Text(if (f.range == "day" && dayLabel != null) "Day: $dayLabel" else "Pick a day...")
                }

                SectionLabel("MORE")
                FilterDropdown("Camera", f.cameraId, cameraOptions, { vm.form = f.copy(cameraId = it) })
                if (f.kind != "person") {
                    Spacer(Modifier.height(6.dp))
                    FilterDropdown("Vehicle type", f.vehicleType, typeOptions, { vm.form = f.copy(vehicleType = it) })
                    Spacer(Modifier.height(6.dp))
                    FilterDropdown("Vehicle colour", f.color, colorOptions, { vm.form = f.copy(color = it) })
                }
                if (f.kind != "vehicle") {
                    Spacer(Modifier.height(6.dp))
                    FilterDropdown("Top colour", f.upperColor, colorOptions, { vm.form = f.copy(upperColor = it) })
                    Spacer(Modifier.height(6.dp))
                    FilterDropdown("Bottom colour", f.lowerColor, colorOptions, { vm.form = f.copy(lowerColor = it) })
                }

                Spacer(Modifier.height(14.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(onClick = { vm.runSearch() }, modifier = Modifier.weight(1f).height(48.dp)) {
                        Text("Search", fontSize = 16.sp)
                    }
                    OutlinedButton(
                        onClick = { vm.form = SearchForm() },
                        modifier = Modifier.height(48.dp),
                    ) { Text("Reset") }
                }

                val err = vm.searchError
                if (err != null) {
                    Spacer(Modifier.height(8.dp))
                    Text(err, color = MaterialTheme.colorScheme.error)
                }
                if (vm.searchedOnce && err == null && !vm.searching) {
                    Spacer(Modifier.height(10.dp))
                    Text("${vm.searchTotal} result(s)", fontSize = 13.sp, color = MaterialTheme.colorScheme.secondary)
                }
            }
        }

        itemsIndexed(vm.searchResults, key = { _, e -> e.id }) { index, e ->
            if (index >= vm.searchResults.size - 4 && vm.hasMore) {
                LaunchedEffect(vm.searchResults.size) { vm.loadMore() }
            }
            EventCard(
                e = e,
                thumbUrl = vm.mediaUrl(e.thumbPath),
                nowMs = nowMs,
                highlight = false,
                onClick = { onOpen(e) },
            )
        }

        item {
            if (vm.searching) {
                Row(Modifier.fillMaxWidth().padding(20.dp), horizontalArrangement = Arrangement.Center) {
                    CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
                }
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}
