export type User = { id: string; email: string }

let currentUser: User | null = null
let token: string | null = null

export function hydrateAuth() {
  token = localStorage.getItem("token")
  const rawUser = localStorage.getItem("user")
  currentUser = rawUser ? JSON.parse(rawUser) : null
  return { currentUser, token }
}

export async function login(email: string, password: string, redirectTo?: string) {
  const response = await fetch("/api/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  })
  const data = await response.json()
  token = data.token
  currentUser = data.user
  localStorage.setItem("token", token || "")
  localStorage.setItem("user", JSON.stringify(currentUser))
  window.location.href = redirectTo || "/dashboard"
}

export function logout() {
  token = null
  currentUser = null
  localStorage.removeItem("token")
  localStorage.removeItem("user")
  window.location.href = "/login"
}

export function isAuthenticated() {
  return Boolean(token && currentUser)
}
