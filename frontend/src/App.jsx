import { useState, useRef, useEffect } from "react"

const API = "http://127.0.0.1:8080"

function Message({ msg }) {
  const isUser = msg.role === "user"
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: "16px",
      padding: "0 16px"
    }}>
      <div style={{
        maxWidth: "72%",
        padding: "12px 16px",
        borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        background: isUser ? "#2563eb" : "#f1f5f9",
        color: isUser ? "#fff" : "#0f172a",
        fontSize: "15px",
        lineHeight: "1.6",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word"
      }}>
        {msg.content}
        {msg.streaming && (
          <span style={{
            display: "inline-block",
            width: "8px",
            height: "15px",
            background: "#64748b",
            marginLeft: "2px",
            verticalAlign: "middle",
            animation: "blink 1s step-end infinite"
          }}/>
        )}
      </div>
    </div>
  )
}

function ModelSelector({ models, selected, onChange }) {
  return (
    <select
      value={selected}
      onChange={e => onChange(e.target.value)}
      style={{
        background: "transparent",
        border: "1px solid #334155",
        color: "#94a3b8",
        borderRadius: "8px",
        padding: "6px 12px",
        fontSize: "13px",
        cursor: "pointer"
      }}
    >
      {models.map(m => (
        <option key={m.id} value={m.id}>{m.name} ({m.size_mb}MB)</option>
      ))}
    </select>
  )
}

export default function App() {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState("")
  const [loading, setLoading]     = useState(false)
  const [models, setModels]       = useState([])
  const [model, setModel]         = useState("tinyllama")
  const bottomRef                 = useRef(null)
  const textareaRef               = useRef(null)

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(data => {
        setModels(data)
        if (data.length > 0) setModel(data[0].id)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = async () => {
    if (!input.trim() || loading) return

    const userMsg = { role: "user", content: input.trim() }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput("")
    setLoading(true)

    // Add empty assistant message that we'll stream into
    const assistantMsg = { role: "assistant", content: "", streaming: true }
    setMessages([...newMessages, assistantMsg])

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages,
          model,
          max_tokens: 512,
          temperature: 0.8,
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
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: fullText,
                    streaming: true
                  }
                  return updated
                })
              }
              if (data.status === "complete") {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: fullText,
                    streaming: false
                  }
                  return updated
                })
              }
            } catch {}
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Error connecting to laylow server. Make sure it is running.",
          streaming: false
        }
        return updated
      })
    }

    setLoading(false)
  }

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      background: "#0f172a",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    }}>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        textarea { resize: none; outline: none; }
        select { outline: none; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "16px 24px",
        borderBottom: "1px solid #1e293b",
        background: "#0f172a"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "32px", height: "32px",
            background: "linear-gradient(135deg, #2563eb, #7c3aed)",
            borderRadius: "8px",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "16px", color: "#fff", fontWeight: "700"
          }}>L</div>
          <span style={{ color: "#f1f5f9", fontWeight: "600", fontSize: "18px" }}>
            laylow
          </span>
          <span style={{
            fontSize: "11px", color: "#475569",
            background: "#1e293b", padding: "2px 8px",
            borderRadius: "20px"
          }}>local AI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {models.length > 0 && (
            <ModelSelector models={models} selected={model} onChange={setModel} />
          )}
          <div style={{
            width: "8px", height: "8px",
            borderRadius: "50%",
            background: loading ? "#f59e0b" : "#22c55e"
          }}/>
        </div>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        paddingTop: "24px",
        paddingBottom: "8px"
      }}>
        {messages.length === 0 && (
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: "12px",
            color: "#475569"
          }}>
            <div style={{ fontSize: "48px" }}>🔒</div>
            <p style={{ fontSize: "18px", color: "#94a3b8", fontWeight: "500" }}>
              Private AI, running on your machine
            </p>
            <p style={{ fontSize: "14px", color: "#475569" }}>
              Nothing leaves your device. Ever.
            </p>
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: "16px 24px 24px",
        background: "#0f172a",
        borderTop: "1px solid #1e293b"
      }}>
        <div style={{
          display: "flex",
          gap: "12px",
          alignItems: "flex-end",
          background: "#1e293b",
          borderRadius: "16px",
          padding: "12px 16px",
          border: "1px solid #334155"
        }}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Message laylow..."
            rows={1}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              color: "#f1f5f9",
              fontSize: "15px",
              lineHeight: "1.6",
              maxHeight: "120px",
              overflowY: "auto",
              fontFamily: "inherit"
            }}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            style={{
              width: "36px", height: "36px",
              borderRadius: "10px",
              border: "none",
              background: loading || !input.trim() ? "#334155" : "#2563eb",
              color: "#fff",
              cursor: loading || !input.trim() ? "not-allowed" : "pointer",
              fontSize: "16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "background 0.15s",
              flexShrink: 0
            }}
          >
            ↑
          </button>
        </div>
        <p style={{
          textAlign: "center",
          fontSize: "12px",
          color: "#334155",
          marginTop: "10px"
        }}>
          laylow runs locally — your conversations stay on your device
        </p>
      </div>
    </div>
  )
}