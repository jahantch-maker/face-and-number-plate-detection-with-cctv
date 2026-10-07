package com.gatevision.app

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.ZoneId
import java.time.ZoneOffset

private const val LIVE_INTERVAL_MS = 2000L
private const val LIVE_MAX_ITEMS = 300

/** Turns what the user typed into a full server address (adds http:// and :8080 when sensible). */
fun normalizeUrl(raw: String): String {
    var u = raw.trim()
    if (u.isEmpty()) return ""
    if (!u.startsWith("http://") && !u.startsWith("https://")) u = "http://$u"
    u = u.trimEnd('/')
    val hostPort = u.substringAfter("://").substringBefore("/")
    val isIp = Regex("""\d{1,3}(\.\d{1,3}){3}""").matches(hostPort)
    if (u.startsWith("http://") && isIp) u = "$u:8080"
    return u
}

class GateVm(app: Application) : AndroidViewModel(app) {
    private val prefs = Prefs(app)

    // ------------------------------------------------------------- session
    var session by mutableStateOf<Session?>(prefs.loadSession())
        private set
    var loginError by mutableStateOf<String?>(null)
        private set
    var loginBusy by mutableStateOf(false)
        private set
    val savedUrl: String get() = prefs.lastUrl
    val savedUser: String get() = prefs.lastUser

    var meta by mutableStateOf<Meta?>(null)
        private set

    private fun api(): ApiClient? = session?.let { ApiClient(it.baseUrl, it.token) }

    fun mediaUrl(path: String?): String? = session?.let { if (path == null) null else it.baseUrl + path }

    init {
        if (session != null) loadMeta()
    }

    fun login(rawUrl: String, user: String, pass: String) {
        val url = normalizeUrl(rawUrl)
        if (url.isEmpty() || user.isBlank() || pass.isEmpty()) {
            loginError = "Please fill in the server address, user name and password."
            return
        }
        loginBusy = true
        loginError = null
        viewModelScope.launch {
            try {
                val r = ApiClient(url, null).login(user.trim(), pass)
                val s = Session(url, r.token, r.username, r.role)
                prefs.saveSession(s)
                resetData()
                session = s
                loadMeta()
            } catch (e: ApiException) {
                loginError = e.message
            } finally {
                loginBusy = false
            }
        }
    }

    fun logout(message: String? = null) {
        stopLive()
        prefs.clearToken()
        session = null
        resetData()
        loginError = message
    }

    private fun resetData() {
        live = emptyList()
        liveError = null
        searchResults = emptyList()
        searchTotal = 0
        searchPage = 0
        searchPages = 1
        searchError = null
        cams = emptyList()
        camsError = null
    }

    private fun handle(e: ApiException): String? {
        if (e.code == 401) {
            logout("Session expired - please sign in again.")
            return null
        }
        return e.message
    }

    private fun loadMeta() {
        val api = api() ?: return
        viewModelScope.launch {
            try {
                meta = api.meta()
            } catch (e: ApiException) {
                handle(e)
            }
        }
    }

    // ---------------------------------------------------------------- live
    var live by mutableStateOf<List<Event>>(emptyList())
        private set
    var liveError by mutableStateOf<String?>(null)
        private set
    var lastLiveOkMs by mutableStateOf(0L)
        private set
    var fresh by mutableStateOf<Set<Long>>(emptySet())
        private set

    private var liveJob: Job? = null

    fun startLive() {
        if (session == null || liveJob?.isActive == true) return
        liveJob = viewModelScope.launch {
            while (isActive) {
                pollLive()
                delay(LIVE_INTERVAL_MS)
            }
        }
    }

    fun stopLive() {
        liveJob?.cancel()
        liveJob = null
    }

    private suspend fun pollLive() {
        val api = api() ?: return
        try {
            val after = live.firstOrNull()?.id ?: 0L
            val got = api.events(after, if (after == 0L) 60 else 100)
            if (got.isNotEmpty()) {
                live = (got + live).distinctBy { it.id }.take(LIVE_MAX_ITEMS)
                if (after != 0L) {
                    val ids = got.map { it.id }.toSet()
                    fresh = fresh + ids
                    viewModelScope.launch {
                        delay(6000)
                        fresh = fresh - ids
                    }
                }
            }
            lastLiveOkMs = System.currentTimeMillis()
            liveError = null
        } catch (e: ApiException) {
            liveError = handle(e)
        }
    }

    // -------------------------------------------------------------- search
    var form by mutableStateOf(SearchForm())
    var searchResults by mutableStateOf<List<Event>>(emptyList())
        private set
    var searchTotal by mutableStateOf(0)
        private set
    var searching by mutableStateOf(false)
        private set
    var searchError by mutableStateOf<String?>(null)
        private set
    var searchedOnce by mutableStateOf(false)
        private set
    private var searchPage by mutableStateOf(0)
    private var searchPages by mutableStateOf(1)
    private var searchJob: Job? = null

    private fun timeRange(f: SearchForm): Pair<Long?, Long?> {
        val now = System.currentTimeMillis() / 1000
        return when (f.range) {
            "1h" -> Pair(now - 3600, null)
            "6h" -> Pair(now - 6 * 3600, null)
            "24h" -> Pair(now - 86400, null)
            "7d" -> Pair(now - 7 * 86400, null)
            "30d" -> Pair(now - 30 * 86400, null)
            "day" -> {
                val picked = f.dayMillis
                if (picked == null) {
                    Pair(null, null)
                } else {
                    // the date picker returns UTC midnight of the chosen calendar day
                    val day = Instant.ofEpochMilli(picked).atZone(ZoneOffset.UTC).toLocalDate()
                    val start = day.atStartOfDay(ZoneId.systemDefault()).toEpochSecond()
                    Pair(start, start + 86400)
                }
            }
            else -> Pair(null, null)
        }
    }

    fun runSearch() {
        val api = api() ?: return
        searchJob?.cancel()
        searching = true
        searchError = null
        searchedOnce = true
        searchResults = emptyList()
        searchTotal = 0
        searchPage = 0
        searchPages = 1
        val f = form
        val (from, to) = timeRange(f)
        searchJob = viewModelScope.launch {
            try {
                val r = api.search(f, from, to, 1)
                searchResults = r.events
                searchTotal = r.total
                searchPage = r.page
                searchPages = r.pages
            } catch (e: ApiException) {
                searchError = handle(e)
            } finally {
                searching = false
            }
        }
    }

    fun loadMore() {
        val api = api() ?: return
        if (searching || searchPage == 0 || searchPage >= searchPages) return
        searching = true
        val f = form
        val (from, to) = timeRange(f)
        val next = searchPage + 1
        searchJob = viewModelScope.launch {
            try {
                val r = api.search(f, from, to, next)
                searchResults = (searchResults + r.events).distinctBy { it.id }
                searchPage = r.page
                searchPages = r.pages
            } catch (e: ApiException) {
                searchError = handle(e)
            } finally {
                searching = false
            }
        }
    }

    val hasMore: Boolean get() = searchPage in 1 until searchPages

    fun searchPlate(plate: String) {
        form = SearchForm(plate = plate, range = "30d")
        runSearch()
    }

    // ------------------------------------------------------------- cameras
    var cams by mutableStateOf<List<CamStatus>>(emptyList())
        private set
    var camsError by mutableStateOf<String?>(null)
        private set

    suspend fun refreshCams() {
        val api = api() ?: return
        try {
            cams = api.status()
            camsError = null
        } catch (e: ApiException) {
            camsError = handle(e)
        }
    }
}
