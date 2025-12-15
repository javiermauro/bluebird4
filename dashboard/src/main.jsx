import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// StrictMode temporarily disabled to prevent WebSocket double-mounting in dev
createRoot(document.getElementById('root')).render(
  <App />
)
