import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { Langfuse } from 'langfuse';

const app = express();
app.use(cors());
app.use(express.json());

if (!process.env.OPENAI_API_KEY) {
  console.warn('[warn] OPENAI_API_KEY is not set.');
}
const langfuseEnabled = !!(process.env.LANGFUSE_PUBLIC_KEY && process.env.LANGFUSE_SECRET_KEY);
if (!langfuseEnabled) {
  console.warn('[warn] LANGFUSE keys not set — tracing disabled.');
}

const langfuse = langfuseEnabled
  ? new Langfuse({
      publicKey: process.env.LANGFUSE_PUBLIC_KEY,
      secretKey: process.env.LANGFUSE_SECRET_KEY,
      baseUrl: process.env.LANGFUSE_BASE_URL || 'https://cloud.langfuse.com',
    })
  : null;

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const VALID_SIZES = ['1024x1024', '1536x1024', '1024x1536'];

// ── LangGraph state ─────────────────────────────────────────────────────────

const GraphState = Annotation.Root({
  originalPrompt: Annotation({ reducer: (_, b) => b ?? _ }),
  size: Annotation({ reducer: (_, b) => b ?? _ }),
  analysis: Annotation({ reducer: (_, b) => b ?? _ }),
  refinedPrompt: Annotation({ reducer: (_, b) => b ?? _ }),
  critiqueResult: Annotation({ reducer: (_, b) => b ?? _ }),
  critiqueFeedback: Annotation({ reducer: (_, b) => b ?? _ }),
  retryCount: Annotation({ reducer: (_, b) => b ?? _ }),
  imageUrl: Annotation({ reducer: (_, b) => b ?? _ }),
  elapsed: Annotation({ reducer: (_, b) => b ?? _ }),
});

// ── Graph builder (closes over SSE emitter) ──────────────────────────────────

function buildGraph(emit, trace) {
  function span(name, input) {
    return trace?.span({ name, input }) ?? null;
  }

  // Node 1: analyze
  async function analyze(state) {
    console.log('\n[graph] ▶ NODE: analyze');
    console.log(`[graph]   input: "${state.originalPrompt}"`);
    emit({ type: 'step', label: 'Analyzing your prompt…' });

    const s = span('analyze', { prompt: state.originalPrompt });
    const res = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 150,
      messages: [
        {
          role: 'system',
          content: `You are a creative assistant. Analyze the user's image request in 2-3 short sentences:
- Detect the language and note if translation is needed
- Identify the core subject, mood, and style cues
- Flag any ambiguities or missing visual details
Reply in English only. Be concise.`,
        },
        { role: 'user', content: state.originalPrompt },
      ],
    });

    const analysis = res.choices[0]?.message?.content ?? '';
    console.log(`[graph]   analysis: ${analysis}`);
    s?.end({ output: { analysis }, usage: res.usage });

    emit({ type: 'step_detail', detail: analysis });
    emit({ type: 'step_done' });

    return { analysis };
  }

  // Node 2: refine
  async function refine(state) {
    console.log(`\n[graph] ▶ NODE: refine  (retryCount=${state.retryCount ?? 0})`);
    if (state.critiqueFeedback) {
      console.log(`[graph]   critique feedback: ${state.critiqueFeedback}`);
    }

    const label = state.retryCount > 0
      ? `Refining prompt (attempt ${state.retryCount + 1})…`
      : 'Crafting image prompt…';

    emit({ type: 'step', label });

    const feedbackClause = state.critiqueFeedback
      ? `\n\nPrevious critique: ${state.critiqueFeedback}\nPlease address these issues in your revised prompt.`
      : '';

    const s = span('refine', { analysis: state.analysis, retryCount: state.retryCount, feedback: state.critiqueFeedback });

    let refined = '';
    const stream = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 300,
      stream: true,
      messages: [
        {
          role: 'system',
          content: `You are an expert image prompt engineer for AI image generators.
1. If the prompt is not in English, translate it first.
2. Enhance with specific visual details: lighting, art style, composition, mood, color palette.
Rules: Reply with ONLY the final English prompt — no explanation, no quotes. Under 200 words.${feedbackClause}`,
        },
        {
          role: 'user',
          content: `Analysis: ${state.analysis}\n\nOriginal prompt: ${state.originalPrompt}`,
        },
      ],
    });

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content ?? '';
      if (token) {
        refined += token;
        emit({ type: 'token', text: token });
      }
    }

    console.log(`[graph]   refined prompt: ${refined}`);
    s?.end({ output: { refinedPrompt: refined } });
    emit({ type: 'step_done' });

    return { refinedPrompt: refined };
  }

  // Node 3: critique
  async function critique(state) {
    console.log('\n[graph] ▶ NODE: critique');
    emit({ type: 'step', label: 'Reviewing prompt quality…' });

    const s = span('critique', { refinedPrompt: state.refinedPrompt });
    const res = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 200,
      messages: [
        {
          role: 'system',
          content: `You are a quality reviewer for AI image generation prompts.
Evaluate this prompt on: specificity, visual clarity, style consistency, and composition guidance.
Reply with JSON only: {"result": "good" | "retry", "feedback": "<one sentence if retry, empty string if good>"}
Use "retry" only if the prompt is vague, contradictory, or missing critical visual information.`,
        },
        { role: 'user', content: state.refinedPrompt },
      ],
    });

    let critiqueResult = 'good';
    let critiqueFeedback = '';

    try {
      const raw = res.choices[0]?.message?.content ?? '{}';
      const parsed = JSON.parse(raw.match(/\{[\s\S]*\}/)?.[0] ?? '{}');
      critiqueResult = parsed.result === 'retry' ? 'retry' : 'good';
      critiqueFeedback = parsed.feedback ?? '';
    } catch {
      // parsing failed → treat as good
    }

    console.log(`[graph]   result: ${critiqueResult}${critiqueFeedback ? ` | feedback: ${critiqueFeedback}` : ''}`);
    s?.end({ output: { result: critiqueResult, feedback: critiqueFeedback }, usage: res.usage });

    const detailMsg = critiqueResult === 'good'
      ? 'Prompt approved.'
      : `Needs improvement: ${critiqueFeedback}`;

    emit({ type: 'step_detail', detail: detailMsg });
    emit({ type: 'step_done' });

    return { critiqueResult, critiqueFeedback };
  }

  // Routing after critique
  function shouldRetry(state) {
    const route = (state.critiqueResult === 'retry' && (state.retryCount ?? 0) < 2)
      ? 'refine'
      : 'generate';
    console.log(`\n[graph] ── ROUTE: critique → ${route}`);
    return route;
  }

  // Increment retry counter
  async function incrementRetry(state) {
    const next = (state.retryCount ?? 0) + 1;
    console.log(`[graph]   retryCount: ${state.retryCount ?? 0} → ${next}`);
    return { retryCount: next };
  }

  // Node 4: generate
  async function generate(state) {
    console.log('\n[graph] ▶ NODE: generate');
    console.log(`[graph]   size: ${state.size}`);
    console.log(`[graph]   prompt: ${state.refinedPrompt || state.originalPrompt}`);

    emit({ type: 'step', label: 'Generating image with gpt-image-1…' });

    const s = span('generate', { prompt: state.refinedPrompt, size: state.size });
    const start = Date.now();
    const response = await openai.images.generate({
      model: 'gpt-image-1',
      prompt: state.refinedPrompt || state.originalPrompt,
      size: state.size,
      n: 1,
    });

    const item = response.data?.[0];
    if (!item) throw new Error('No image data returned by OpenAI');

    const imageUrl = item.b64_json
      ? `data:image/png;base64,${item.b64_json}`
      : item.url;

    if (!imageUrl) throw new Error('No image data in OpenAI response');

    const elapsed = Date.now() - start;
    console.log(`[graph]   image generated in ${elapsed}ms`);
    s?.end({ output: { elapsed } });
    emit({ type: 'step_done' });

    return { imageUrl, elapsed };
  }

  const graph = new StateGraph(GraphState)
    .addNode('analyze', analyze)
    .addNode('refine', refine)
    .addNode('critique', critique)
    .addNode('incrementRetry', incrementRetry)
    .addNode('generate', generate)
    .addEdge(START, 'analyze')
    .addEdge('analyze', 'refine')
    .addEdge('refine', 'critique')
    .addConditionalEdges('critique', shouldRetry, {
      refine: 'incrementRetry',
      generate: 'generate',
    })
    .addEdge('incrementRetry', 'refine')
    .addEdge('generate', END);

  return graph.compile();
}

// ── SSE helper ────────────────────────────────────────────────────────────────

function sse(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// ── Route ─────────────────────────────────────────────────────────────────────

app.post('/api/generate', async (req, res) => {
  const { prompt, size = '1024x1024' } = req.body;

  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'Prompt is required' });
  }
  if (!VALID_SIZES.includes(size)) {
    return res.status(400).json({ error: `Size must be one of: ${VALID_SIZES.join(', ')}` });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const start = Date.now();
  const emit = (data) => sse(res, data);

  // LangFuse trace
  const trace = langfuse?.trace({
    name: 'image-generation',
    input: { prompt, size },
    sessionId: crypto.randomUUID(),
  }) ?? null;

  console.log(`\n${'─'.repeat(50)}`);
  console.log(`[generate] START  prompt="${prompt}"  size=${size}`);

  try {
    const compiled = buildGraph(emit, trace);

    const finalState = await compiled.invoke({
      originalPrompt: prompt,
      size,
      analysis: '',
      refinedPrompt: '',
      critiqueResult: '',
      critiqueFeedback: '',
      retryCount: 0,
      imageUrl: '',
      elapsed: 0,
    });

    const elapsed = Date.now() - start;
    console.log(`\n[generate] DONE  elapsed=${elapsed}ms  retries=${finalState.retryCount}`);
    console.log('─'.repeat(50));

    trace?.update({ output: { refinedPrompt: finalState.refinedPrompt, elapsed, retries: finalState.retryCount } });
    await langfuse?.flushAsync();

    sse(res, {
      type: 'result',
      url: finalState.imageUrl,
      elapsed,
      refinedPrompt: finalState.refinedPrompt !== prompt ? finalState.refinedPrompt : undefined,
    });
  } catch (err) {
    console.error('[generate] ERROR:', err?.message ?? err);
    trace?.update({ output: { error: err?.message } });
    await langfuse?.flushAsync();
    sse(res, { type: 'error', message: err?.message ?? 'Image generation failed' });
  }

  res.write('data: [DONE]\n\n');
  res.end();
});

// Serve built frontend in production
if (process.env.NODE_ENV === 'production') {
  const distPath = join(__dirname, '../dist');
  app.use(express.static(distPath));
  app.get('*', (_, res) => res.sendFile(join(distPath, 'index.html')));
}

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`Image generation API → http://localhost:${port}`);
});
