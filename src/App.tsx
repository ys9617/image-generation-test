import { useCallback, useEffect, useRef, useState } from 'react';

interface ThinkingStep {
  label: string;
  tokens: string;
  detail?: string;
  done: boolean;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text?: string;
  imageUrl?: string;
  elapsedMs?: number;
  refinedPrompt?: string;
  thinking?: ThinkingStep[];
  error?: boolean;
}

type ImageSize = '1024x1024' | '1536x1024' | '1024x1536';

const SIZE_OPTIONS: { label: string; value: ImageSize }[] = [
  { label: 'Square', value: '1024x1024' },
  { label: 'Wide', value: '1536x1024' },
  { label: 'Tall', value: '1024x1536' },
];

function ThinkingPanel({ steps }: { steps: ThinkingStep[] }) {
  return (
    <div className="thinking-panel">
      {steps.map((step, i) => (
        <div key={i} className={`thinking-step ${step.done ? 'done' : 'active'}`}>
          <span className="step-icon">{step.done ? '✓' : '●'}</span>
          <div className="step-body">
            <span className="step-label">{step.label}</span>
            {step.detail && <p className="step-detail">{step.detail}</p>}
            {step.tokens && <p className="step-tokens">{step.tokens}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [liveSteps, setLiveSteps] = useState<ThinkingStep[]>([]);
  const [size, setSize] = useState<ImageSize>('1024x1024');
  const [copyFeedback, setCopyFeedback] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const shared = params.get('prompt');
    if (shared) setPrompt(shared);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, liveSteps, loading]);

  const handleGenerate = useCallback(async () => {
    const trimmed = prompt.trim();
    if (!trimmed || loading) return;

    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', text: trimmed }]);
    setLiveSteps([]);
    setLoading(true);

    const steps: ThinkingStep[] = [];

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: trimmed, size }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6);
          if (raw === '[DONE]') break;

          const event = JSON.parse(raw);

          if (event.type === 'step') {
            steps.push({ label: event.label, tokens: '', done: false });
            setLiveSteps([...steps]);
          }

          if (event.type === 'token') {
            steps[steps.length - 1].tokens += event.text;
            setLiveSteps([...steps]);
          }

          if (event.type === 'step_detail') {
            steps[steps.length - 1].detail = event.detail;
            setLiveSteps([...steps]);
          }

          if (event.type === 'step_done') {
            steps[steps.length - 1].done = true;
            setLiveSteps([...steps]);
          }

          if (event.type === 'result') {
            setMessages(prev => [...prev, {
              id: crypto.randomUUID(),
              role: 'assistant',
              imageUrl: event.url,
              text: trimmed,
              elapsedMs: event.elapsed,
              refinedPrompt: event.refinedPrompt,
              thinking: [...steps],
            }]);
            setLiveSteps([]);
          }

          if (event.type === 'error') {
            setMessages(prev => [...prev, {
              id: crypto.randomUUID(),
              role: 'assistant',
              text: event.message,
              error: true,
            }]);
            setLiveSteps([]);
          }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: err instanceof Error ? err.message : 'Failed to generate image.',
        error: true,
      }]);
      setLiveSteps([]);
    } finally {
      setLoading(false);
    }
  }, [prompt, size, loading]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  }, [handleGenerate]);

  const handleShare = async () => {
    const url = new URL(window.location.href);
    url.searchParams.set('prompt', prompt.trim());
    await navigator.clipboard.writeText(url.toString());
    setCopyFeedback(true);
    setTimeout(() => setCopyFeedback(false), 2000);
  };

  const handleDownload = (imageUrl: string, label: string) => {
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `${label.slice(0, 40).replace(/[^a-z0-9]/gi, '_') || 'image'}.png`;
    a.click();
  };

  const shouldDisable = !prompt.trim() || loading;

  return (
    <div className="page-shell">
      <header className="topbar">
        <div>
          <h1>Image Agent</h1>
          <p className="subtitle">gpt-4o-mini → gpt-image-1</p>
        </div>
        <button className="secondary" onClick={handleShare} disabled={!prompt.trim()}>
          {copyFeedback ? '✓ Copied!' : 'Share prompt'}
        </button>
      </header>

      <main>
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <p>Describe an image and press <kbd>⌘ Enter</kbd> to generate.</p>
          </div>
        )}

        <section className="chat-panel">
          {messages.map(msg => (
            <div key={msg.id} className={`bubble ${msg.role}${msg.error ? ' error' : ''}`}>
              {msg.text && <p className="bubble-text">{msg.text}</p>}

              {msg.thinking && msg.thinking.length > 0 && (
                <details className="thinking-details">
                  <summary>Thought for {((msg.elapsedMs ?? 0) / 1000).toFixed(1)}s</summary>
                  <ThinkingPanel steps={msg.thinking} />
                </details>
              )}

              {msg.refinedPrompt && (
                <p className="optimized-prompt">✦ {msg.refinedPrompt}</p>
              )}

              {msg.imageUrl && (
                <div className="image-card">
                  <img src={msg.imageUrl} alt={msg.text ?? 'Generated image'} />
                  <div className="image-overlay">
                    <button
                      className="download-btn"
                      onClick={() => handleDownload(msg.imageUrl!, msg.text ?? 'image')}
                    >
                      ↓ Download
                    </button>
                    {msg.elapsedMs != null && (
                      <span className="gen-time">{(msg.elapsedMs / 1000).toFixed(1)}s</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Live thinking panel while generating */}
          {loading && (
            <div className="bubble assistant">
              {liveSteps.length > 0
                ? <ThinkingPanel steps={liveSteps} />
                : <div className="loading-dots"><span /><span /><span /></div>
              }
            </div>
          )}

          <div ref={chatEndRef} />
        </section>

        <section className="prompt-box">
          <div className="size-picker">
            {SIZE_OPTIONS.map(opt => (
              <button
                key={opt.value}
                className={`size-btn${size === opt.value ? ' active' : ''}`}
                onClick={() => setSize(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe the image you want to create… (any language)"
            rows={3}
          />

          <div className="actions">
            <button className="primary" onClick={handleGenerate} disabled={shouldDisable}>
              {loading ? 'Generating…' : 'Generate'}
            </button>
            {messages.length > 0 && (
              <button className="secondary" onClick={() => setMessages([])}>
                Clear
              </button>
            )}
            <span className="shortcut-hint">⌘ Return</span>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
