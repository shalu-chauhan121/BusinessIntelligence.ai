/**
 * Firebase initialisation.
 *
 * Every value comes from `VITE_FIREBASE_*` environment variables. When they are
 * absent the module reports `configured: false` and the app falls back to the
 * backend's demo login, so the product is runnable before a Firebase project
 * exists. See docs/FIREBASE_SETUP.md.
 */
import { initializeApp, getApps } from 'firebase/app'
import {
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  updateProfile,
} from 'firebase/auth'

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
}

export const firebaseConfigured = Boolean(config.apiKey && config.projectId && config.appId)

let auth = null
if (firebaseConfigured) {
  const app = getApps().length ? getApps()[0] : initializeApp(config)
  auth = getAuth(app)
}

export { auth }

export const googleProvider = firebaseConfigured ? new GoogleAuthProvider() : null

export function watchAuth(callback) {
  if (!auth) return () => {}
  return onAuthStateChanged(auth, callback)
}

export async function firebaseSignUp(email, password, displayName) {
  const cred = await createUserWithEmailAndPassword(auth, email, password)
  if (displayName) await updateProfile(cred.user, { displayName })
  return cred.user
}

export async function firebaseSignIn(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password)
  return cred.user
}

export async function firebaseGoogleSignIn() {
  const cred = await signInWithPopup(auth, googleProvider)
  return cred.user
}

export async function firebaseSignOut() {
  if (auth) await signOut(auth)
}

/** Friendly text for the error codes users actually hit. */
export function describeAuthError(error) {
  const code = error?.code || ''
  const map = {
    'auth/invalid-email': 'That email address does not look right.',
    'auth/user-not-found': 'No account exists for that email address.',
    'auth/wrong-password': 'Incorrect password.',
    'auth/invalid-credential': 'Those credentials were not accepted.',
    'auth/email-already-in-use': 'An account already exists for that email address.',
    'auth/weak-password': 'Choose a password of at least 6 characters.',
    'auth/popup-closed-by-user': 'The Google sign-in window was closed before finishing.',
    'auth/operation-not-allowed':
      'This sign-in method is not enabled in the Firebase console (Authentication → Sign-in method).',
    'auth/network-request-failed': 'Could not reach Firebase. Check your network connection.',
  }
  return map[code] || error?.message || 'Authentication failed.'
}
