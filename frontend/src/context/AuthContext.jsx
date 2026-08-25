/**
 * Authentication + authorisation state.
 *
 * Authentication is Firebase (email/password or Google). Authorisation is ours:
 * the role the user chose at sign-up lives in our database and is returned by
 * `/api/auth/me` together with the permission set the backend will enforce.
 *
 * When Firebase is not configured the provider transparently uses the backend's
 * demo login instead, so the product can be run and reviewed before keys exist.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api, setTokenProvider } from '../lib/api'
import {
  auth as firebaseAuth,
  describeAuthError,
  firebaseConfigured,
  firebaseGoogleSignIn,
  firebaseSignIn,
  firebaseSignOut,
  firebaseSignUp,
  watchAuth,
} from '../lib/firebase'

const AuthContext = createContext(null)
const DEMO_TOKEN_KEY = 'bi-demo-token'

export function AuthProvider({ children }) {
  const [status, setStatus] = useState('loading') // loading | anonymous | authenticated
  const [profile, setProfile] = useState(null)
  const [authMode, setAuthMode] = useState(firebaseConfigured ? 'firebase' : 'demo')
  const [notice, setNotice] = useState(null)
  const demoToken = useRef(
    typeof window !== 'undefined' ? window.localStorage.getItem(DEMO_TOKEN_KEY) : null,
  )
  const firebaseUser = useRef(null)

  // one token source for every API call
  useEffect(() => {
    setTokenProvider(async () => {
      if (firebaseConfigured && firebaseUser.current) return firebaseUser.current.getIdToken()
      return demoToken.current
    })
  }, [])

  const loadProfile = useCallback(async () => {
    const me = await api.me()
    setProfile(me)
    setStatus('authenticated')
    return me
  }, [])

  // learn which mode the backend is in (it may verify tokens or not)
  useEffect(() => {
    api
      .authConfig()
      .then((cfg) => {
        setAuthMode(firebaseConfigured ? 'firebase' : cfg.mode)
        setNotice(cfg.notice)
      })
      .catch(() => setNotice('The API is not reachable. Start the backend and reload.'))
  }, [])

  // firebase session
  useEffect(() => {
    if (!firebaseConfigured) return undefined
    return watchAuth(async (user) => {
      firebaseUser.current = user
      if (!user) {
        setProfile(null)
        setStatus('anonymous')
        return
      }
      try {
        await loadProfile()
      } catch {
        setStatus('anonymous')
      }
    })
  }, [loadProfile])

  // demo session
  useEffect(() => {
    if (firebaseConfigured) return
    if (!demoToken.current) {
      setStatus('anonymous')
      return
    }
    loadProfile().catch(() => {
      demoToken.current = null
      window.localStorage.removeItem(DEMO_TOKEN_KEY)
      setStatus('anonymous')
    })
  }, [loadProfile])

  const signUp = useCallback(
    async ({ email, password, displayName, role, organisation }) => {
      if (firebaseConfigured) {
        await firebaseSignUp(email, password, displayName)
        await api.register({ role, display_name: displayName, organisation })
        return loadProfile()
      }
      const res = await api.demoLogin({ email, display_name: displayName, role })
      demoToken.current = res.token
      window.localStorage.setItem(DEMO_TOKEN_KEY, res.token)
      return loadProfile()
    },
    [loadProfile],
  )

  const signIn = useCallback(
    async ({ email, password }) => {
      if (firebaseConfigured) {
        await firebaseSignIn(email, password)
        return loadProfile()
      }
      const res = await api.demoLogin({ email, display_name: '', role: 'business_leader' })
      demoToken.current = res.token
      window.localStorage.setItem(DEMO_TOKEN_KEY, res.token)
      return loadProfile()
    },
    [loadProfile],
  )

  const signInWithGoogle = useCallback(
    async (role = 'business_leader') => {
      if (!firebaseConfigured) throw new Error('Google sign-in requires Firebase configuration.')
      const user = await firebaseGoogleSignIn()
      await api.register({ role, display_name: user.displayName || '', organisation: '' })
      return loadProfile()
    },
    [loadProfile],
  )

  const logout = useCallback(async () => {
    if (firebaseConfigured) await firebaseSignOut()
    demoToken.current = null
    window.localStorage.removeItem(DEMO_TOKEN_KEY)
    firebaseUser.current = null
    setProfile(null)
    setStatus('anonymous')
  }, [])

  const changeRole = useCallback(async (role) => {
    const updated = await api.setRole(role)
    setProfile(updated)
    return updated
  }, [])

  // Presentation only — changes how findings are framed and what is
  // recommended, never what the server is willing to send. That stays on role.
  const changePersona = useCallback(async (persona) => {
    const updated = await api.setPersona(persona)
    setProfile(updated)
    return updated
  }, [])

  const value = useMemo(
    () => ({
      status,
      profile,
      role: profile?.role || null,
      persona: profile?.persona || null,
      can: (permission) => Boolean(profile?.permissions?.[permission]),
      isAnalyst: profile?.role === 'data_analyst',
      authMode,
      firebaseConfigured,
      notice,
      signUp,
      signIn,
      signInWithGoogle,
      logout,
      changeRole,
      changePersona,
      describeAuthError,
    }),
    [status, profile, authMode, notice, signUp, signIn, signInWithGoogle, logout, changeRole,
     changePersona],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
