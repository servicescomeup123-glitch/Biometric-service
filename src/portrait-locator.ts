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
    console.log('[Portrait] Téléchargement det_10g.onnx...');
    const modelBuffer = await downloadModel(MODEL_URLS.detection);
    console.log('[Portrait] Modèle téléchargé, taille:', modelBuffer.byteLength, 'bytes');
    detector = await ort.InferenceSession.create(modelBuffer, {
      executionProviders:     ['cpu'],
      graphOptimizationLevel: 'all',
    });
    console.log('[Portrait] ✓ Modèle chargé avec succès');
    console.log('[Portrait] Inputs:', JSON.stringify(detector.inputNames));
    console.log('[Portrait] Outputs:', JSON.stringify(detector.outputNames));
  } catch (err) {
    console.error('[Portrait] ✗ Échec chargement modèle:', err);
    detector = null;
  }
  return detector;
}

// ─── Types ───────────────────────────────────────────────────────────────────

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
    if (zone && zone.confidence > 0.5) {
      console.log('[Portrait] ONNX réussi - confiance:', zone.confidence.toFixed(2));
      return zone;
    }
  }

  console.log('[Portrait] Fallback heuristique');
  return heuristicPortraitZone(warpedBuffer, originalW, originalH);
}

// ─── Détection ONNX (SCRFD det_10g) ──────────────────────────────────────────
//
// Outputs confirmés par run test :
//   "448" → [12800,1]  scores stride 8  (80×80×2 anchors)
//   "471" → [3200,1]   scores stride 16 (40×40×2 anchors)
//   "494" → [800,1]    scores stride 32 (20×20×2 anchors)
//   "451" → [12800,4]  bbox   stride 8
//   "474" → [3200,4]   bbox   stride 16
//   "497" → [800,4]    bbox   stride 32
//   "454","477","500"  keypoints — ignorés

const SCRFD_STRIDE_MAP = [
  { stride: 8,  N: 12800, scoreOut: '448', bboxOut: '451' },
  { stride: 16, N: 3200,  scoreOut: '471', bboxOut: '474' },
  { stride: 32, N: 800,   scoreOut: '494', bboxOut: '497' },
];
const INPUT_SIZE  = 640;
const NUM_ANCHORS = 2;

async function detectWithONNX(
  session: any,
  imageBuffer: Buffer,
  W: number,
  H: number,
): Promise<PortraitZone | null> {
  try {
    const ort = await getOrtModule();

    const resized = await sharp(imageBuffer)
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'fill', kernel: 'lanczos3' })
      .removeAlpha()
      .raw()
      .toBuffer();

    // Normalisation InsightFace : (pixel - 127.5) / 128
    const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      tensor[i]                               = (resized[i * 3]     - 127.5) / 128.0;
      tensor[INPUT_SIZE * INPUT_SIZE + i]     = (resized[i * 3 + 1] - 127.5) / 128.0;
      tensor[2 * INPUT_SIZE * INPUT_SIZE + i] = (resized[i * 3 + 2] - 127.5) / 128.0;
    }

    const inputName   = session.inputNames[0]; // "input.1"
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const results     = await session.run({ [inputName]: inputTensor });

    let topScore = 0;
    let topBox: { x1: number; y1: number; x2: number; y2: number } | null = null;

    for (const { stride, N, scoreOut, bboxOut } of SCRFD_STRIDE_MAP) {
      const scores = results[scoreOut]?.data as Float32Array | undefined;
      const bboxes = results[bboxOut]?.data  as Float32Array | undefined;

      if (!scores || !bboxes) {
        console.warn(`[Portrait ONNX] Stride ${stride}: outputs "${scoreOut}"/"${bboxOut}" manquants`);
        continue;
      }

      if (scores.length !== N || bboxes.length !== N * 4) {
        console.warn(`[Portrait ONNX] Stride ${stride}: taille inattendue scores=${scores.length} bboxes=${bboxes.length}`);
        continue;
      }

      const fH = Math.floor(INPUT_SIZE / stride);
      const fW = Math.floor(INPUT_SIZE / stride);

      // Centres d'anchors selon InsightFace officiel :
      // les NUM_ANCHORS anchors d'une même cellule partagent le même centre
      for (let i = 0; i < N; i++) {
        const score = scores[i];
        if (score < 0.4 || score <= topScore) continue;

        const cellIdx = Math.floor(i / NUM_ANCHORS);
        const cx_cell = cellIdx % fW;
        const cy_cell = Math.floor(cellIdx / fW);

        // Centre identique pour tous les anchors de cette cellule
        const ax = (cx_cell + 0.5) * stride;
        const ay = (cy_cell + 0.5) * stride;

        // SCRFD : les deltas bbox sont déjà en pixels (pas besoin de * stride)
        const x1 = (ax - bboxes[i * 4 + 0]) / INPUT_SIZE * W;
        const y1 = (ay - bboxes[i * 4 + 1]) / INPUT_SIZE * H;
        const x2 = (ax + bboxes[i * 4 + 2]) / INPUT_SIZE * W;
        const y2 = (ay + bboxes[i * 4 + 3]) / INPUT_SIZE * H;

        topScore = score;
        topBox   = { x1, y1, x2, y2 };
      }
    }

    if (!topBox) {
      console.log('[Portrait ONNX] Aucun visage détecté (score max < 0.4)');
      return null;
    }

    const boxW   = topBox.x2 - topBox.x1;
    const boxH   = topBox.y2 - topBox.y1;
    const margin = Math.min(boxW, boxH) * 0.25;

    const left   = Math.max(0, Math.floor(topBox.x1 - margin));
    const top    = Math.max(0, Math.floor(topBox.y1 - margin));
    const width  = Math.min(W - left, Math.ceil(boxW + 2 * margin));
    const height = Math.min(H - top,  Math.ceil(boxH + 2 * margin));

    console.log(
      `[Portrait ONNX] ✓ Visage — score: ${topScore.toFixed(3)}` +
      ` zone: [${left},${top} ${width}×${height}]`,
    );

    return { left, top, width, height, confidence: topScore, method: 'onnx_face_detection' };

  } catch (err) {
    console.error('[Portrait ONNX] Erreur:', err);
    return null;
  }
}

// ─── Heuristique documentaire (fallback) ─────────────────────────────────────

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
        const mean        = sum / raw.length;
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