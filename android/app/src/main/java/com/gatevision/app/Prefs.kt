package com.gatevision.app

import android.content.Context

data class Session(val baseUrl: String, val token: String, val username: String, val role: String)

class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("gatevision", Context.MODE_PRIVATE)

    var lastUrl: String
        get() = sp.getString("url", "") ?: ""
        set(v) {
            sp.edit().putString("url", v).apply()
        }

    var lastUser: String
        get() = sp.getString("user", "") ?: ""
        set(v) {
            sp.edit().putString("user", v).apply()
        }

    fun loadSession(): Session? {
        val url = sp.getString("url", "") ?: ""
        val token = sp.getString("token", "") ?: ""
        if (url.isEmpty() || token.isEmpty()) return null
        return Session(url, token, sp.getString("user", "") ?: "", sp.getString("role", "") ?: "")
    }

    fun saveSession(s: Session) {
        sp.edit()
            .putString("url", s.baseUrl)
            .putString("token", s.token)
            .putString("user", s.username)
            .putString("role", s.role)
            .apply()
    }

    fun clearToken() {
        sp.edit().remove("token").remove("role").apply()
    }
}
