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
    console.log('[Portrait] Inputs:', detector.inputNames);
    console.log('[Portrait] Outputs:', detector.outputNames);
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
    if (zone && zone.confidence > 0.5) {
      console.log('[Portrait] ONNX réussi - confiance:', zone.confidence.toFixed(2));
      return zone;
    }
  }

  console.log('[Portrait] Fallback heuristique');
  return heuristicPortraitZone(warpedBuffer, originalW, originalH);
}

// ─── Détection ONNX (SCRFD / det_10g) ────────────────────────────────────────

async function detectWithONNX(
  session: any,
  imageBuffer: Buffer,
  W: number,
  H: number,
): Promise<PortraitZone | null> {
  try {
    const ort = await getOrtModule();

    // det_10g est entraîné sur 640×640 — taille critique
    const INPUT_SIZE = 640;

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

    const inputName   = session.inputNames[0];
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    console.log('[Portrait ONNX] Run — input:', inputName,
      '| outputs:', session.outputNames.join(', '));

    const results = await session.run({ [inputName]: inputTensor });

    // ── Parser les outputs SCRFD multi-échelle ────────────────────────────────
    // det_10g produit 6 outputs pour les strides 8, 16, 32 :
    //   score_stride_X  : [1, H*W*2, 1]  — probabilité de visage
    //   bbox_stride_X   : [1, H*W*2, 4]  — régression bbox (distance aux 4 côtés)
    //
    // En pratique les outputNames sont dans l'ordre :
    //   [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32]   (ordre InsightFace)
    // ou numérotés différemment selon l'export — on les identifie par leur taille.

    const strides     = [8, 16, 32];
    const numAnchors  = 2; // SCRFD 10G : 2 anchors par cellule

    let topScore = 0;
    let topBox: { x1: number; y1: number; x2: number; y2: number } | null = null;

    for (const stride of strides) {
      const fH = Math.floor(INPUT_SIZE / stride);
      const fW = Math.floor(INPUT_SIZE / stride);
      const N  = fH * fW * numAnchors; // nombre de détections pour ce stride

      // Trouver l'output de score pour ce stride (shape [N] ou [N,1] ou [1,N,1])
      const scoreOut = session.outputNames.find((name: string) => {
        const d = results[name]?.data as Float32Array | undefined;
        return d && d.length === N;
      });

      // Trouver l'output de bbox pour ce stride (shape [N*4] ou [N,4] ou [1,N,4])
      const bboxOut = session.outputNames.find((name: string) => {
        const d = results[name]?.data as Float32Array | undefined;
        return d && d.length === N * 4;
      });

      if (!scoreOut || !bboxOut) {
        console.log(`[Portrait ONNX] Stride ${stride}: outputs non trouvés`);
        continue;
      }

      const scores = results[scoreOut].data as Float32Array;
      const bboxes = results[bboxOut].data  as Float32Array;

      // Générer les centres d'anchors pour ce stride
      const anchorCenters: Array<[number, number]> = [];
      for (let y = 0; y < fH; y++) {
        for (let x = 0; x < fW; x++) {
          for (let a = 0; a < numAnchors; a++) {
            anchorCenters.push([(x + 0.5) * stride, (y + 0.5) * stride]);
          }
        }
      }

      for (let i = 0; i < N; i++) {
        const score = scores[i];
        if (score < 0.4 || score <= topScore) continue;

        const [ax, ay] = anchorCenters[i];

        // SCRFD bbox = distance depuis le centre de l'anchor
        const x1 = (ax - bboxes[i * 4 + 0] * stride) / INPUT_SIZE * W;
        const y1 = (ay - bboxes[i * 4 + 1] * stride) / INPUT_SIZE * H;
        const x2 = (ax + bboxes[i * 4 + 2] * stride) / INPUT_SIZE * W;
        const y2 = (ay + bboxes[i * 4 + 3] * stride) / INPUT_SIZE * H;

        topScore = score;
        topBox   = { x1, y1, x2, y2 };
      }
    }

    if (!topBox) {
      console.log('[Portrait ONNX] Aucun visage détecté (score max < seuil)');
      return null;
    }

    const boxW   = topBox.x2 - topBox.x1;
    const boxH   = topBox.y2 - topBox.y1;
    const margin = Math.min(boxW, boxH) * 0.25;

    const left   = Math.max(0, Math.floor(topBox.x1 - margin));
    const top    = Math.max(0, Math.floor(topBox.y1 - margin));
    const width  = Math.min(W - left, Math.ceil(boxW + 2 * margin));
    const height = Math.min(H - top,  Math.ceil(boxH + 2 * margin));

    console.log(`[Portrait ONNX] Visage trouvé — score: ${topScore.toFixed(3)}`,
      `box: [${left},${top},${width}x${height}]`);

    return { left, top, width, height, confidence: topScore, method: 'onnx_face_detection' };

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