import sharp from 'sharp';

// ─── Configuration R2 ────────────────────────────────────────────────────────

const R2_MODELS_BASE = process.env.R2_MODELS_BASE_URL ||
  'https://pub-91a604b2df2f4a17b8aa07c2c2eee859.r2.dev/models/buffalo_l';

const MODEL_URLS = {
  arcface: `${R2_MODELS_BASE}/w600k_mbf.onnx`,
};

let arcfaceSession: any           = null;
let ortModule: any                = null;
let onnxAvailable: boolean | null = null;

// ─── Chargement ONNX ─────────────────────────────────────────────────────────

async function getOrtModule(): Promise<any> {
  if (ortModule) return ortModule;
  ortModule = require('onnxruntime-node');
  return ortModule;
}

async function downloadModel(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`HTTP ${response.status} — ${url}`);
  return response.arrayBuffer();
}

export async function getArcFaceSession(): Promise<any | null> {
  if (arcfaceSession) return arcfaceSession;
  if (onnxAvailable === false) return null;

  try {
    const ort         = await getOrtModule();
    console.log('[ArcFace] Téléchargement modèle depuis R2...');
    const modelBuffer = await downloadModel(MODEL_URLS.arcface);

    arcfaceSession = await ort.InferenceSession.create(modelBuffer, {
      executionProviders:     ['cpu'],
      graphOptimizationLevel: 'all',
    });

    onnxAvailable = true;
    console.log('[ArcFace] Session ONNX créée avec succès');
    return arcfaceSession;
  } catch (err) {
    onnxAvailable = false;
    console.error('[ArcFace] ONNX indisponible:', err);
    return null;
  }
}

export type FaceEmbedding = Float32Array;

// ─── Génération d'embedding ───────────────────────────────────────────────────

export async function generateEmbedding(
  portraitBuffer: Buffer,
): Promise<{ embedding: FaceEmbedding | null; confidence: number; error?: string; method?: string }> {

  const session = await getArcFaceSession();

  if (session) {
    try {
      const ort = await getOrtModule();

      // Préprocessing InsightFace officiel :
      // 1. Resize 112×112 (taille native ArcFace)
      // 2. PAS de normalize() sharp — ça casserait la distribution d'entraînement
      // 3. Normalisation pixel : (pixel / 127.5) - 1.0  → valeurs dans [-1, +1]
      // Note : sharpen léger OK car il ne change pas la distribution globale
      const resized = await sharp(portraitBuffer)
        .resize(112, 112, { fit: 'fill', kernel: 'lanczos3' })
        .removeAlpha()
        .raw()
        .toBuffer();

      // Format NCHW : [1, 3, 112, 112]
      // Canal R → index i, G → 112*112+i, B → 2*112*112+i
      const tensor = new Float32Array(3 * 112 * 112);
      for (let i = 0; i < 112 * 112; i++) {
        tensor[i]                 = (resized[i * 3]     / 127.5) - 1.0; // R
        tensor[112 * 112 + i]     = (resized[i * 3 + 1] / 127.5) - 1.0; // G
        tensor[2 * 112 * 112 + i] = (resized[i * 3 + 2] / 127.5) - 1.0; // B
      }

      const inputName   = session.inputNames[0];  // 'input.1'
      const outputName  = session.outputNames[0]; // '516'
      const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 112, 112]);
      const results     = await session.run({ [inputName]: inputTensor });

      const embedData = results[outputName]?.data as Float32Array;
      if (!embedData || embedData.length !== 512) {
        throw new Error(`Embedding invalide: length=${embedData?.length ?? 'null'}`);
      }

      console.log('[ArcFace] ✓ Embedding 512-dim généré');
      return { embedding: l2Normalize(embedData), confidence: 0.95, method: 'onnx' };

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn('[ArcFace] Erreur ONNX:', msg, '— bascule Claude Vision');
      onnxAvailable = false;
    }
  }

  // ── Fallback Claude Vision ─────────────────────────────────────────────────
  console.log('[ArcFace] Fallback Claude Vision pour embedding');
  return generateClaudeEmbedding(portraitBuffer);
}

// ─── Pseudo-embedding Claude Vision (fallback) ────────────────────────────────

async function generateClaudeEmbedding(
  portraitBuffer: Buffer,
): Promise<{ embedding: FaceEmbedding | null; confidence: number; error?: string; method?: string }> {
  try {
    const base64 = portraitBuffer.toString('base64');

    const prompt = `Tu es un système d'analyse biométrique faciale. Analyse ce portrait et extrait 16 descripteurs faciaux normalisés entre -1.0 et 1.0.

Ces descripteurs doivent capturer les caractéristiques stables (structure osseuse, proportions) et non les attributs variables (expression, éclairage).

Réponds UNIQUEMENT avec un tableau JSON de 16 nombres flottants entre -1.0 et 1.0, rien d'autre.
Exemple : [-0.23, 0.45, 0.12, -0.67, 0.89, -0.34, 0.56, 0.78, -0.12, 0.34, -0.89, 0.67, 0.23, -0.45, 0.56, -0.78]

Descripteurs à encoder (dans cet ordre) :
1. Largeur relative du visage (étroit=-1, large=+1)
2. Ratio hauteur/largeur du visage
3. Position verticale des yeux
4. Écartement relatif des yeux
5. Taille relative du nez
6. Largeur des narines
7. Épaisseur des lèvres
8. Ratio hauteur/largeur de la bouche
9. Proéminence des pommettes
10. Angle de la mâchoire
11. Proéminence du menton
12. Hauteur du front
13. Symétrie faciale globale
14. Ratio nez/visage
15. Position des sourcils
16. Profondeur des orbites`;

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type':    'application/json',
        'x-api-key':       process.env.ANTHROPIC_API_KEY!,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model:      'claude-sonnet-4-5',
        max_tokens: 256,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: base64 } },
            { type: 'text',  text: prompt },
          ],
        }],
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('[Claude Embedding] Erreur API:', err);
      return { embedding: null, confidence: 0, error: 'Claude API error', method: 'claude' };
    }

    const result = await response.json() as any;
    const text   = (result.content?.[0]?.text || '').trim();

    const arrayMatch = text.match(/\[[\s\S]*\]/);
    if (!arrayMatch) {
      console.error('[Claude Embedding] Pas de tableau trouvé:', text);
      return { embedding: null, confidence: 0, error: 'Parse error', method: 'claude' };
    }

    const descriptors: number[] = JSON.parse(arrayMatch[0]);
    if (descriptors.length !== 16) {
      return { embedding: null, confidence: 0, error: 'Mauvaise dimension', method: 'claude' };
    }

    const embedding = l2Normalize(new Float32Array(descriptors));
    console.log('[Claude Embedding] Descripteurs extraits avec succès (16-dim)');
    return { embedding, confidence: 0.70, method: 'claude' };

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[Claude Embedding] Erreur:', msg);
    return { embedding: null, confidence: 0, error: msg, method: 'claude' };
  }
}

// ─── L2 normalisation ────────────────────────────────────────────────────────

export function l2Normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm < 1e-10) return vec;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i] / norm;
  return result;
}