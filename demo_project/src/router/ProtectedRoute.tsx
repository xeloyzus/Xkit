import { isAuthenticated, hydrateAuth } from "../auth/useAuth"

export function ProtectedRoute({ children }: { children: any }) {
  hydrateAuth()
  if (!isAuthenticated()) {
    window.location.href = "/login"
    return null
  }
  return children
}
