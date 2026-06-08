import { hydrateAuth, isAuthenticated } from "../src/auth/useAuth"

test("user stays authenticated after refresh", () => {
  localStorage.setItem("token", "abc")
  localStorage.setItem("user", JSON.stringify({ id: "1", email: "a@example.com" }))
  hydrateAuth()
  expect(isAuthenticated()).toBe(true)
})
