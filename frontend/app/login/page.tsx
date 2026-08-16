"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

const API_URL = "https://drugmicrobeai.onrender.com";

export default function LoginPage() {
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "recovery">("login");

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [recoveryCode, setRecoveryCode] = useState("");
  const [newPassword, setNewPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setError("");
    setMessage("");

    if (!username.trim() || !password) {
      setError("Username and password are required.");
      return;
    }

    try {
      setLoading(true);

      const response = await fetch(
        `${API_URL}/auth/login`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            username: username.trim(),
            password,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Login failed."
        );
      }

      localStorage.setItem(
        "drugMicrobeAuthToken",
        data.token
      );

      localStorage.setItem(
        "drugMicrobeAuthUser",
        JSON.stringify(data.user)
      );

      router.push("/");
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Login failed."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleRecovery(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setError("");
    setMessage("");

    if (
      !username.trim() ||
      !recoveryCode.trim() ||
      !newPassword
    ) {
      setError(
        "Username, recovery code and new password are required."
      );
      return;
    }

    try {
      setLoading(true);

      const response = await fetch(
        `${API_URL}/auth/password/reset`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            username: username.trim(),
            recovery_code: recoveryCode.trim(),
            new_password: newPassword,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Password reset failed."
        );
      }

      setMessage(
        "Password reset successfully. You can now sign in."
      );

      setMode("login");
      setPassword("");
      setRecoveryCode("");
      setNewPassword("");
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Password reset failed."
      );
    } finally {
      setLoading(false);
    }
  }

  const isRecovery = mode === "recovery";

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-white">
      <div className="mx-auto flex min-h-[80vh] max-w-md items-center justify-center">
        <div className="w-full rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
            DrugMicrobe AI
          </p>

          <h1 className="mt-3 text-3xl font-bold">
            {isRecovery
              ? "Reset your password"
              : "Welcome back"}
          </h1>

          <p className="mt-2 text-sm text-slate-400">
            {isRecovery
              ? "Use your recovery code to create a new password."
              : "Sign in to continue to the research dashboard."}
          </p>

          {isRecovery ? (
            <form
              onSubmit={handleRecovery}
              className="mt-8 space-y-5"
            >
              <div>
                <label className="text-sm text-slate-300">
                  Username
                </label>

                <input
                  value={username}
                  onChange={(event) =>
                    setUsername(event.target.value)
                  }
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-cyan-400"
                  placeholder="Enter username"
                  autoComplete="username"
                />
              </div>

              <div>
                <label className="text-sm text-slate-300">
                  Recovery code
                </label>

                <input
                  value={recoveryCode}
                  onChange={(event) =>
                    setRecoveryCode(event.target.value)
                  }
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-cyan-400"
                  placeholder="Enter recovery code"
                  autoComplete="off"
                />
              </div>

              <div>
                <label className="text-sm text-slate-300">
                  New password
                </label>

                <input
                  type="password"
                  value={newPassword}
                  onChange={(event) =>
                    setNewPassword(event.target.value)
                  }
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-cyan-400"
                  placeholder="At least 6 characters"
                  autoComplete="new-password"
                />
              </div>

              {error && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                  {error}
                </div>
              )}

              {message && (
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-cyan-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading
                  ? "Resetting..."
                  : "Reset password"}
              </button>

              <button
                type="button"
                onClick={() => {
                  setMode("login");
                  setError("");
                  setMessage("");
                }}
                className="w-full text-sm font-semibold text-slate-400 transition hover:text-white"
              >
                Back to sign in
              </button>
            </form>
          ) : (
            <form
              onSubmit={handleLogin}
              className="mt-8 space-y-5"
            >
              <div>
                <label className="text-sm text-slate-300">
                  Username
                </label>

                <input
                  value={username}
                  onChange={(event) =>
                    setUsername(event.target.value)
                  }
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-cyan-400"
                  placeholder="Enter username"
                  autoComplete="username"
                />
              </div>

              <div>
                <label className="text-sm text-slate-300">
                  Password
                </label>

                <input
                  type="password"
                  value={password}
                  onChange={(event) =>
                    setPassword(event.target.value)
                  }
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-cyan-400"
                  placeholder="Enter password"
                  autoComplete="current-password"
                />
              </div>

              {error && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                  {error}
                </div>
              )}

              {message && (
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-cyan-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Signing in..." : "Sign in"}
              </button>

              <button
                type="button"
                onClick={() => {
                  setMode("recovery");
                  setError("");
                  setMessage("");
                }}
                className="w-full text-sm font-semibold text-cyan-400 transition hover:text-cyan-300"
              >
                Forgot password?
              </button>
            </form>
          )}

          <div className="mt-6 text-center text-sm text-slate-500">
            {isRecovery ? (
              <>
                Don't have an account?{" "}
                <a
                  href="/signup"
                  className="font-semibold text-cyan-400 hover:text-cyan-300"
                >
                  Create account
                </a>
              </>
            ) : (
              <>
                Don't have an account?{" "}
                <a
                  href="/signup"
                  className="font-semibold text-cyan-400 hover:text-cyan-300"
                >
                  Create account
                </a>
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
