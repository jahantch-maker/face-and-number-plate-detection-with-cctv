package com.gatevision.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ApiException(val code: Int, message: String) : Exception(message)

data class Event(
    val id: Long,
    val ts: Double,
    val cameraName: String,
    val direction: String,
    val kind: String,
    val plateText: String?,
    val plateConf: Double?,
    val vehicleType: String?,
    val color: String?,
    val upperColor: String?,
    val lowerColor: String?,
    val fullPath: String?,
    val cropPath: String?,
    val platePath: String?,
    val facePath: String?,
) {
    val isVehicle: Boolean get() = kind == "vehicle"

    /** Small picture shown in lists: plate for vehicles, face for people. */
    val thumbPath: String?
        get() = if (isVehicle) (platePath ?: cropPath ?: fullPath) else (facePath ?: cropPath ?: fullPath)
}

data class Camera(val id: String, val name: String, val direction: String, val role: String)
data class Meta(val cameras: List<Camera>, val vehicleTypes: List<String>, val colors: List<String>)
data class CamStatus(
    val id: String,
    val name: String,
    val role: String,
    val direction: String,
    val online: Boolean,
    val ageSeconds: Int?,
    val fps: Double?,
)

data class SearchResult(val events: List<Event>, val total: Int, val page: Int, val pages: Int)
data class LoginResult(val token: String, val username: String, val role: String)

data class SearchForm(
    val plate: String = "",
    val fuzzy: Boolean = false,
    val kind: String = "",
    val direction: String = "",
    val cameraId: String = "",
    val vehicleType: String = "",
    val color: String = "",
    val upperColor: String = "",
    val lowerColor: String = "",
    val range: String = "24h",      // 1h, 6h, 24h, 7d, 30d, all, day
    val dayMillis: Long? = null,    // for range == "day": the picked date (UTC midnight, from the date picker)
)

private fun JSONObject.str(key: String): String? =
    if (isNull(key) || !has(key)) null else optString(key).ifBlank { null }

private fun JSONObject.dbl(key: String): Double? =
    if (isNull(key) || !has(key)) null else optDouble(key)

private fun parseEvent(o: JSONObject) = Event(
    id = o.getLong("id"),
    ts = o.optDouble("ts", 0.0),
    cameraName = o.optString("camera_name", ""),
    direction = o.optString("direction", ""),
    kind = o.optString("kind", ""),
    plateText = o.str("plate_text"),
    plateConf = o.dbl("plate_conf"),
    vehicleType = o.str("vehicle_type"),
    color = o.str("color"),
    upperColor = o.str("upper_color"),
    lowerColor = o.str("lower_color"),
    fullPath = o.str("full_path"),
    cropPath = o.str("crop_path"),
    platePath = o.str("plate_path"),
    facePath = o.str("face_path"),
)

private fun parseEvents(a: JSONArray?): List<Event> {
    if (a == null) return emptyList()
    val out = ArrayList<Event>(a.length())
    for (i in 0 until a.length()) out.add(parseEvent(a.getJSONObject(i)))
    return out
}

private fun parseStrings(a: JSONArray?): List<String> {
    if (a == null) return emptyList()
    val out = ArrayList<String>(a.length())
    for (i in 0 until a.length()) out.add(a.getString(i))
    return out
}

class ApiClient(private val baseUrl: String, private val token: String?) {

    companion object {
        val http: OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .writeTimeout(20, TimeUnit.SECONDS)
            .build()
    }

    fun mediaUrl(path: String?): String? = if (path == null) null else baseUrl + path

    private suspend fun call(
        path: String,
        query: Map<String, String> = emptyMap(),
        jsonBody: JSONObject? = null,
    ): JSONObject = withContext(Dispatchers.IO) {
        try {
            val url = (baseUrl + path).toHttpUrl().newBuilder().apply {
                for ((k, v) in query) addQueryParameter(k, v)
            }.build()
            val builder = Request.Builder().url(url)
            if (token != null) builder.header("Authorization", "Bearer $token")
            if (jsonBody != null) {
                builder.post(jsonBody.toString().toRequestBody("application/json".toMediaType()))
            } else {
                builder.get()
            }
            http.newCall(builder.build()).execute().use { resp ->
                val text = resp.body?.string() ?: ""
                val json = try {
                    JSONObject(text)
                } catch (e: Exception) {
                    JSONObject()
                }
                if (!resp.isSuccessful) {
                    throw ApiException(resp.code, json.optString("error", "Server answered HTTP ${resp.code}"))
                }
                json
            }
        } catch (e: ApiException) {
            throw e
        } catch (e: IllegalArgumentException) {
            throw ApiException(0, "That server address is not valid.")
        } catch (e: IOException) {
            throw ApiException(0, "Cannot reach the server. Is Tailscale connected? (${e.message ?: "network error"})")
        }
    }

    suspend fun login(username: String, password: String): LoginResult {
        val body = JSONObject().put("username", username).put("password", password)
        val r = call("/api/v1/login", jsonBody = body)
        return LoginResult(r.getString("token"), r.optString("username", username), r.optString("role", ""))
    }

    suspend fun meta(): Meta {
        val r = call("/api/v1/meta")
        val cams = ArrayList<Camera>()
        val arr = r.optJSONArray("cameras")
        if (arr != null) {
            for (i in 0 until arr.length()) {
                val c = arr.getJSONObject(i)
                cams.add(Camera(c.optString("id"), c.optString("name"), c.optString("direction"), c.optString("role")))
            }
        }
        return Meta(cams, parseStrings(r.optJSONArray("vehicle_types")), parseStrings(r.optJSONArray("colors")))
    }

    /** Newest first. With afterId > 0 only events newer than that id are returned. */
    suspend fun events(afterId: Long, limit: Int): List<Event> {
        val r = call("/api/v1/events", mapOf("after_id" to afterId.toString(), "limit" to limit.toString()))
        return parseEvents(r.optJSONArray("events"))
    }

    suspend fun search(form: SearchForm, tsFrom: Long?, tsTo: Long?, page: Int): SearchResult {
        val q = LinkedHashMap<String, String>()
        fun put(k: String, v: String) {
            if (v.isNotBlank()) q[k] = v
        }
        put("plate", form.plate.trim())
        if (form.fuzzy) q["fuzzy"] = "1"
        put("kind", form.kind)
        put("direction", form.direction)
        put("camera_id", form.cameraId)
        put("vehicle_type", form.vehicleType)
        put("color", form.color)
        put("upper_color", form.upperColor)
        put("lower_color", form.lowerColor)
        if (tsFrom != null) q["ts_from"] = tsFrom.toString()
        if (tsTo != null) q["ts_to"] = tsTo.toString()
        q["page"] = page.toString()
        q["page_size"] = "40"
        val r = call("/api/v1/search", q)
        return SearchResult(
            parseEvents(r.optJSONArray("events")),
            r.optInt("total", 0),
            r.optInt("page", page),
            r.optInt("pages", 1),
        )
    }

    suspend fun status(): List<CamStatus> {
        val r = call("/api/v1/status")
        val out = ArrayList<CamStatus>()
        val arr = r.optJSONArray("cameras") ?: return out
        for (i in 0 until arr.length()) {
            val c = arr.getJSONObject(i)
            out.add(
                CamStatus(
                    id = c.optString("camera_id"),
                    name = c.optString("name"),
                    role = c.optString("role"),
                    direction = c.optString("direction"),
                    online = c.optBoolean("online", false),
                    ageSeconds = if (c.isNull("age")) null else c.optInt("age"),
                    fps = c.dbl("fps"),
                ),
            )
        }
        return out
    }
}
