import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { Spinner } from './components/ui'
import { useAuth } from './context/AuthContext'
import AuthPage from './pages/AuthPage'
import Dashboard from './pages/Dashboard'
import DataPage from './pages/DataPage'
import DocumentsPage from './pages/DocumentsPage'
import HistoryPage from './pages/HistoryPage'
import InvestigationPage from './pages/InvestigationPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  const { status } = useAuth()

  if (status === 'loading') {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Starting BusinessIntelligence.ai…" />
      </div>
    )
  }

  if (status === 'anonymous') {
    return (
      <Routes>
        <Route path="/signup" element={<AuthPage mode="signup" />} />
        <Route path="*" element={<AuthPage mode="signin" />} />
      </Routes>
    )
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/investigation" element={<InvestigationPage />} />
        <Route path="/investigation/:investigationId" element={<InvestigationPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/data" element={<DataPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Layout>
  )
}
