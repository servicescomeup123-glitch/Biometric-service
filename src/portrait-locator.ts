import sharp from 'sharp';

// ─── Configuration R2 ────────────────────────────────────────────────────────

const R2_MODELS_BASE = process.env.R2_MODELS_BASE_URL ||
  'https://pub-91a604b2df2f4a17b8aa07c2c2eee859.r2.dev/models/buffalo_l';

const MODEL_URLS = {
  detection: `${R2_MODELS_BASE}/det_10g.onnx`,
};

let detector: any  = null;
let ortModule: any = null;

// ─── Chargement ONNX ─────────────────────────────────────────────────────────

async function getOrtModule(): Promise<any> {
  if (ortModule) return ortModule;
  ortModule = require('onnxruntime-node');
  console.log('[Portrait] Module ONNX chargé');
  return ortModule;
}

async function downloadModel(url: string): Promise<ArrayBuffer> {
  console.log(`[Portrait] Téléchargement depuis R2: ${url}`);
  const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.arrayBuffer();
}

export async function getDetector(): Promise<any | null> {
  if (detector) return detector;
  try {
    const ort         = await getOrtModule();
    const modelBuffer = await downloadModel(MODEL_URLS.detection);
    detector = await ort.InferenceSession.create(modelBuffer, {
      executionProviders:     ['cpu'],
      graphOptimizationLevel: 'all',
    });
    console.log('[Portrait] Modèle chargé');
  } catch (err) {
    console.warn('[Portrait] Modèle indisponible, fallback heuristique:', err);
  }
  return detector;
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PortraitZone {
  left:       number;
  top:        number;
  width:      number;
  height:     number;
  confidence: number;
  method:     'onnx_face_detection' | 'heuristic_zone' | 'full_document';
}

// ─── Localisation du portrait ─────────────────────────────────────────────────

export async function locatePortrait(
  warpedBuffer: Buffer,
  originalW: number,
  originalH: number,
): Promise<PortraitZone> {

  const session = await getDetector();
  if (session) {
    const zone = await detectWithONNX(session, warpedBuffer, originalW, originalH);
    if (zone && zone.confidence > 0.6) {
      console.log('[Portrait] ONNX réussi - confiance:', zone.confidence.toFixed(2));
      return zone;
    }
  }

  console.log('[Portrait] Fallback heuristique');
  return heuristicPortraitZone(warpedBuffer, originalW, originalH);
}

// ─── Détection ONNX ───────────────────────────────────────────────────────────

async function detectWithONNX(
  session: any,
  imageBuffer: Buffer,
  W: number,
  H: number,
): Promise<PortraitZone | null> {
  try {
    const ort = await getOrtModule();

    const resized = await sharp(imageBuffer)
      .resize(128, 128, { fit: 'fill' })
      .raw()
      .toBuffer();

    const tensor = new Float32Array(3 * 128 * 128);
    for (let i = 0; i < 128 * 128; i++) {
      tensor[i]                 = resized[i * 3]     / 255.0;
      tensor[128 * 128 + i]     = resized[i * 3 + 1] / 255.0;
      tensor[2 * 128 * 128 + i] = resized[i * 3 + 2] / 255.0;
    }

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 128, 128]);
    const results = await session.run({ "input.1": inputTensor });

    const boxes = results['boxes']?.data as Float32Array | undefined;
    if (!boxes || boxes.length < 5) return null;

    const confidence = boxes[4];
    if (confidence < 0.5) return null;

    const x1 = Math.floor(boxes[0] * W);
    const y1 = Math.floor(boxes[1] * H);
    const x2 = Math.floor(boxes[2] * W);
    const y2 = Math.floor(boxes[3] * H);

    const margin = Math.min((x2 - x1) * 0.15, (y2 - y1) * 0.15);

    return {
      left:       Math.max(0, x1 - margin),
      top:        Math.max(0, y1 - margin),
      width:      Math.min(W, x2 - x1 + 2 * margin),
      height:     Math.min(H, y2 - y1 + 2 * margin),
      confidence,
      method:     'onnx_face_detection',
    };
  } catch (err) {
    console.error('[Portrait ONNX] Erreur:', err);
    return null;
  }
}

// ─── Heuristique documentaire ─────────────────────────────────────────────────

async function heuristicPortraitZone(
  imageBuffer: Buffer,
  W: number,
  H: number,
): Promise<PortraitZone> {

  const GRID  = 10;
  const cellW = Math.floor(W / GRID);
  const cellH = Math.floor(H / GRID);

  const variances: number[][] = [];

  for (let gy = 0; gy < GRID; gy++) {
    variances[gy] = [];
    for (let gx = 0; gx < GRID; gx++) {
      try {
        const raw = await sharp(imageBuffer)
          .extract({ left: gx * cellW, top: gy * cellH, width: cellW, height: cellH })
          .grayscale()
          .raw()
          .toBuffer();

        let sum = 0, sumSq = 0;
        for (const v of raw) { sum += v; sumSq += v * v; }
        const mean     = sum / raw.length;
        variances[gy][gx] = sumSq / raw.length - mean * mean;
      } catch {
        variances[gy][gx] = 0;
      }
    }
  }

  let maxVariance = 0, bestGx = 0, bestGy = 0;
  for (let gy = 0; gy < Math.ceil(GRID * 0.6); gy++) {
    for (let gx = 0; gx < Math.ceil(GRID * 0.35); gx++) {
      if (variances[gy]?.[gx] > maxVariance) {
        maxVariance = variances[gy][gx];
        bestGx = gx;
        bestGy = gy;
      }
    }
  }

  if (maxVariance === 0) {
    return { left: 0, top: 0, width: W, height: H, confidence: 0.2, method: 'full_document' };
  }

  const portraitLeft   = Math.max(0, (bestGx - 1) * cellW);
  const portraitTop    = Math.max(0, (bestGy - 1) * cellH);
  const portraitWidth  = Math.min(W - portraitLeft, 5 * cellW);
  const portraitHeight = Math.min(H - portraitTop,  5 * cellH);
  const ratio          = portraitHeight / portraitWidth;

  return {
    left:       portraitLeft,
    top:        portraitTop,
    width:      portraitWidth,
    height:     portraitHeight,
    confidence: (ratio >= 0.6 && ratio <= 1.5) ? 0.55 : 0.35,
    method:     'heuristic_zone',
  };
}
