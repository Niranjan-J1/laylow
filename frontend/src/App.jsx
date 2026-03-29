import { useState, useRef, useEffect } from "react"

const API = "http://127.0.0.1:8080"

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: #060608;
    min-height: 100vh;
  }

  .app {
    display: flex;
    height: 100vh;
    background: #060608;
    position: relative;
    overflow: hidden;
  }

  .orb1 {
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    top: -200px; left: -100px;
    pointer-events: none;
  }

  .orb2 {
    position: fixed;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(168,85,247,0.1) 0%, transparent 70%);
    bottom: -150px; right: -100px;
    pointer-events: none;
  }

  .sidebar {
    width: 260px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 20px 12px;
    gap: 8px;
    z-index: 1;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
  }

  .logo-mark {
    width: 32px; height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700; color: #fff;
    box-shadow: 0 0 20px rgba(99,102,241,0.4);
  }

  .logo-text {
    font-size: 17px;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.3px;
  }

  .logo-badge {
    font-size: 10px;
    color: #6366f1;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    padding: 2px 7px;
    border-radius: 20px;
    margin-left: auto;
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px dashed rgba(255,255,255,0.12);
    background: transparent;
    color: rgba(255,255,255,0.4);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
    width: 100%;
  }

  .new-chat-btn:hover {
    border-color: rgba(99,102,241,0.5);
    color: rgba(255,255,255,0.7);
    background: rgba(99,102,241,0.08);
  }

  .section-label {
    font-size: 10px;
    font-weight: 600;
    color: rgba(255,255,255,0.2);
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 12px;
    margin-top: 8px;
  }

  .chat-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 13px;
    color: rgba(255,255,255,0.45);
    border: 1px solid transparent;
  }

  .chat-item:hover {
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.7);
  }

  .chat-item.active {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.25);
    color: rgba(255,255,255,0.85);
  }

  .chat-item-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: rgba(99,102,241,0.6);
    flex-shrink: 0;
  }

  .sidebar-footer {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
  }

  .model-select-wrap {
    padding: 8px 12px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
  }

  .model-select-label {
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
  }

  .model-select {
    width: 100%;
    background: transparent;
    border: none;
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    outline: none;
    appearance: none;
  }

  .model-select option {
    background: #1e1e2e;
    color: #fff;
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    z-index: 1;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(6,6,8,0.8);
    backdrop-filter: blur(12px);
  }

  .topbar-title {
    font-size: 14px;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
  }

  .status-dot {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: rgba(255,255,255,0.3);
  }

  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34,197,94,0.6);
  }

  .dot.loading {
    background: #f59e0b;
    box-shadow: 0 0 8px rgba(245,158,11,0.6);
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .settings-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.4);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }

  .settings-btn:hover {
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.7);
    border-color: rgba(255,255,255,0.15);
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 32px 0;
    scroll-behavior: smooth;
  }

  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }

  .message-wrap {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 28px;
    margin-bottom: 24px;
  }

  .message-wrap.user { display: flex; justify-content: flex-end; }

  .bubble {
    max-width: 72%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .bubble.user {
    background: linear-gradient(135deg, rgba(99,102,241,0.9), rgba(139,92,246,0.9));
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.3);
  }

  .bubble.assistant {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.85);
    border-bottom-left-radius: 4px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }

  .cursor {
    display: inline-block;
    width: 2px;
    height: 16px;
    background: rgba(255,255,255,0.6);
    margin-left: 2px;
    vertical-align: middle;
    animation: blink 0.8s step-end infinite;
  }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    text-align: center;
    padding: 40px;
  }

  .empty-icon {
    width: 64px; height: 64px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(99,102,241,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    margin-bottom: 8px;
    box-shadow: 0 0 40px rgba(99,102,241,0.15);
  }

  .empty h2 {
    font-size: 22px;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    letter-spacing: -0.4px;
  }

  .empty p {
    font-size: 14px;
    color: rgba(255,255,255,0.3);
    max-width: 360px;
    line-height: 1.6;
  }

  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 8px;
  }

  .suggestion {
    padding: 8px 16px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.45);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }

  .suggestion:hover {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.4);
    color: rgba(255,255,255,0.75);
  }

  .input-area {
    padding: 16px 28px 24px;
    max-width: 760px;
    margin: 0 auto;
    width: 100%;
  }

  .input-box {
    display: flex;
    align-items: flex-end;
    gap: 12px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 14px 16px;
    transition: border-color 0.2s;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
  }

  .input-box:focus-within {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(99,102,241,0.2), inset 0 1px 0 rgba(255,255,255,0.06);
  }

  .input-box textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: rgba(255,255,255,0.9);
    font-size: 15px;
    line-height: 1.6;
    font-family: inherit;
    resize: none;
    max-height: 140px;
    overflow-y: auto;
    placeholder-color: rgba(255,255,255,0.2);
  }

  .input-box textarea::placeholder { color: rgba(255,255,255,0.2); }

  .send-btn {
    width: 38px; height: 38px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: #fff;
    cursor: pointer;
    font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
  }

  .send-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(99,102,241,0.5); }
  .send-btn:active { transform: translateY(0); }
  .send-btn:disabled { background: rgba(255,255,255,0.08); box-shadow: none; cursor: not-allowed; transform: none; }

  .input-hint {
    font-size: 11px;
    color: rgba(255,255,255,0.15);
    text-align: center;
    margin-top: 10px;
  }

  .settings-panel {
    position: absolute;
    top: 60px; right: 20px;
    width: 280px;
    background: rgba(15,15,25,0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    z-index: 100;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }

  .settings-title {
    font-size: 13px;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    margin-bottom: 16px;
    letter-spacing: 0.2px;
  }

  .setting-row {
    margin-bottom: 16px;
  }

  .setting-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-bottom: 8px;
  }

  .setting-value {
    color: rgba(255,255,255,0.7);
    font-weight: 500;
  }

  .setting-slider {
    width: 100%;
    accent-color: #6366f1;
    cursor: pointer;
  }
`

const SUGGESTIONS = [
  "Explain quantum computing simply",
  "Write a Python function to sort a list",
  "What causes northern lights?",
  "Help me write a cover letter"
]

export default function App() {
  const [chats, setChats]         = useState([{ id: 1, title: "New chat", messages: [] }])
  const [activeChat, setActiveChat] = useState(1)
  const [input, setInput]         = useState("")
  const [loading, setLoading]     = useState(false)
  const [models, setModels]       = useState([])
  const [model, setModel]         = useState("tinyllama")
  const [showSettings, setShowSettings] = useState(false)
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [temperature, setTemperature]   = useState(0.8)
  const [maxTokens, setMaxTokens]       = useState(512)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  const currentMessages = chats.find(c => c.id === activeChat)?.messages || []

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(data => { setModels(data); if (data.length) setModel(data[0].id) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [currentMessages])

  const updateMessages = (msgs) => {
    setChats(prev => prev.map(c =>
      c.id === activeChat
        ? { ...c, messages: msgs, title: msgs[0]?.content.slice(0, 30) || "New chat" }
        : c
    ))
  }

  const newChat = () => {
    const id = Date.now()
    setChats(prev => [...prev, { id, title: "New chat", messages: [] }])
    setActiveChat(id)
  }

  const send = async (text) => {
    const content = (text || input).trim()
    if (!content || loading) return

    const userMsg = { role: "user", content }
    const newMsgs = [...currentMessages, userMsg]
    updateMessages(newMsgs)
    setInput("")
    setLoading(true)

    const assistantMsg = { role: "assistant", content: "", streaming: true }
    updateMessages([...newMsgs, assistantMsg])

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMsgs,
          model,
          max_tokens: maxTokens,
          temperature,
          top_p: 0.95,
          stream: true
        })
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let fullText = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.token) {
                fullText += data.token
                setChats(prev => prev.map(c =>
                  c.id === activeChat
                    ? { ...c, messages: c.messages.map((m, i) =>
                        i === c.messages.length - 1
                          ? { ...m, content: fullText, streaming: true }
                          : m
                      )}
                    : c
                ))
              }
              if (data.status === "complete") {
                setChats(prev => prev.map(c =>
                  c.id === activeChat
                    ? { ...c, messages: c.messages.map((m, i) =>
                        i === c.messages.length - 1
                          ? { ...m, content: fullText, streaming: false }
                          : m
                      )}
                    : c
                ))
              }
            } catch {}
          }
        }
      }
    } catch {
      setChats(prev => prev.map(c =>
        c.id === activeChat
          ? { ...c, messages: c.messages.map((m, i) =>
              i === c.messages.length - 1
                ? { ...m, content: "Connection error. Is the laylow server running?", streaming: false }
                : m
            )}
          : c
      ))
    }
    setLoading(false)
  }

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() }
  }

    const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch(`${API}/documents`, {
        method: "POST",
        body: formData
      })
      const data = await res.json()
      setUploadedDocs(prev => {
        if (prev.find(d => d.doc_id === data.doc_id)) return prev
        return [...prev, { ...data, filename: file.name }]
      })
    } catch {
      alert("Failed to upload document. Is the server running?")
    }
  }

  return (
    <>
      <style>{STYLES}</style>
      <div className="app">
        <div className="orb1" /><div className="orb2" />

        {/* Sidebar */}
        <div className="sidebar">
          <div className="logo">
            <div className="logo-mark">L</div>
            <span className="logo-text">laylow</span>
            <span className="logo-badge">local</span>
          </div>

          <button className="new-chat-btn" onClick={newChat}>
            <span>+</span> New conversation
          </button>

          {chats.filter(c => c.messages.length > 0).length > 0 && (
            <>
              <div className="section-label">Recent</div>
              {chats.filter(c => c.messages.length > 0).slice().reverse().map(c => (
                <div
                  key={c.id}
                  className={`chat-item ${c.id === activeChat ? "active" : ""}`}
                  onClick={() => setActiveChat(c.id)}
                >
                  <div className="chat-item-dot" />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {c.title}
                  </span>
                </div>
              ))}
            </>
          )}

          <div className="sidebar-footer">
            <div className="model-select-wrap">
              <div className="model-select-label">Model</div>
              <select className="model-select" value={model} onChange={e => setModel(e.target.value)}>
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Main */}
        <div className="main">
          <div className="topbar">
            <span className="topbar-title">
              {chats.find(c => c.id === activeChat)?.title || "New conversation"}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <div className="status-dot">
                <div className={`dot ${loading ? "loading" : ""}`} />
                {loading ? "Thinking..." : "Ready"}
              </div>
              <button className="settings-btn" onClick={() => setShowSettings(s => !s)}>
                ⚙ Settings
              </button>
            </div>
          </div>

          {showSettings && (
            <div className="settings-panel">
              <div className="settings-title">Generation settings</div>
              <div className="setting-row">
                <div className="setting-label">
                  Temperature <span className="setting-value">{temperature.toFixed(1)}</span>
                </div>
                <input type="range" className="setting-slider"
                  min="0" max="2" step="0.1"
                  value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} />
              </div>
              <div className="setting-row">
                <div className="setting-label">
                  Max tokens <span className="setting-value">{maxTokens}</span>
                </div>
                <input type="range" className="setting-slider"
                  min="64" max="2048" step="64"
                  value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} />
              </div>
            </div>
          )}

          <div className="messages" onClick={() => setShowSettings(false)}>
            {currentMessages.length === 0 ? (
              <div className="empty">
                <div className="empty-icon">🔒</div>
                <h2>Private AI on your machine</h2>
                <p>Your conversations never leave your device. No cloud, no tracking, no subscriptions.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map(s => (
                    <button key={s} className="suggestion" onClick={() => send(s)}>{s}</button>
                  ))}
                </div>
              </div>
            ) : (
              currentMessages.map((msg, i) => (
                <div key={i} className={`message-wrap ${msg.role}`}>
                  <div className={`bubble ${msg.role}`}>
                    {msg.content}
                    {msg.streaming && <span className="cursor" />}
                  </div>
                </div>
              ))
            )}
            <div ref={bottomRef} />
          </div>

          <div className="input-area">
            {uploadedDocs.length > 0 && (
              <div style={{
                display: "flex", gap: "8px", marginBottom: "10px",
                flexWrap: "wrap"
              }}>
                {uploadedDocs.map(doc => (
                  <div key={doc.doc_id} style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "4px 10px",
                    background: "rgba(99,102,241,0.12)",
                    border: "1px solid rgba(99,102,241,0.3)",
                    borderRadius: "20px",
                    fontSize: "12px",
                    color: "rgba(255,255,255,0.6)"
                  }}>
                    📄 {doc.filename} ({doc.chunks} chunks)
                  </div>
                ))}
              </div>
            )}
            <div className="input-box">
              <label style={{ cursor: "pointer", color: "rgba(255,255,255,0.25)",
                fontSize: "18px", lineHeight: 1, flexShrink: 0,
                transition: "color 0.15s" }}
                title="Upload document for RAG"
                onMouseOver={e => e.target.style.color = "rgba(255,255,255,0.6)"}
                onMouseOut={e => e.target.style.color = "rgba(255,255,255,0.25)"}
              >
                📎
                <input type="file" accept=".pdf,.txt,.md"
                  style={{ display: "none" }}
                  onChange={handleFileUpload}
                />
              </label>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => {
                  setInput(e.target.value)
                  e.target.style.height = "auto"
                  e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"
                }}
                onKeyDown={handleKey}
                placeholder={uploadedDocs.length > 0
                  ? "Ask about your documents..."
                  : "Ask anything — it stays private"}
                rows={1}
              />
              <button className="send-btn" onClick={() => send()}
                disabled={loading || !input.trim()}>
                ↑
              </button>
            </div>
            <p className="input-hint">Enter to send · Shift+Enter for new line · runs 100% locally</p>
          </div>
        </div>
      </div>
    </>
  )
}