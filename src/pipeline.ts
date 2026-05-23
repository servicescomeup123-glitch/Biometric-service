import sharp from 'sharp';
import { generateEmbedding }        from './face-embedder';
import { locatePortrait }           from './portrait-locator';
import { matchFaces }               from './face-matcher';

export interface BiometricPipelineInput {
  documentImageBase64: string;
  selfieBase64:        string;
}

export interface BiometricPipelineResult {
  matched:           boolean;
  similarityScore:   number;
  riskLevel:         'low' | 'medium' | 'high';
  livenessConfirmed: boolean;
  needsManualReview: boolean;
  failureReason?:    string;
  debug?: {
    docDetectionConfidence:      number;
    portraitDetectionMethod:     string;
    portraitDetectionConfidence: number;
    selfieDetectionMethod:       string;
    selfieDetectionConfidence:   number;
    embeddingMethod:             string;
    durationMs:                  number;
  };
}

// ─── Normalisation image ──────────────────────────────────────────────────────

async function normalizeInput(
  base64: string,
): Promise<{ buffer: Buffer; width: number; height: number }> {
  const data   = base64.replace(/^data:image\/\w+;base64,/, '');
  const buffer = Buffer.from(data, 'base64');
  const meta   = await sharp(buffer).metadata();
  const width  = meta.width  ?? 800;
  const height = meta.height ?? 600;

  if (width > 1600 || height > 1600) {
    const resized = await sharp(buffer)
      .resize(1600, 1600, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 90 })
      .toBuffer();
    const rm = await sharp(resized).metadata();
    return { buffer: resized, width: rm.width ?? width, height: rm.height ?? height };
  }

  return { buffer, width, height };
}

// ─── Crop + resize vers 112×112 pour ArcFace ─────────────────────────────────

async function cropAndResize(
  imageBuffer: Buffer,
  zone: { left: number; top: number; width: number; height: number },
): Promise<Buffer> {
  return sharp(imageBuffer)
    .extract({
      left:   Math.max(0, Math.round(zone.left)),
      top:    Math.max(0, Math.round(zone.top)),
      width:  Math.max(1, Math.round(zone.width)),
      height: Math.max(1, Math.round(zone.height)),
    })
    .resize(112, 112, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .jpeg({ quality: 95 })
    .toBuffer();
}

// ─── Détection document ───────────────────────────────────────────────────────

async function detectDocument(buffer: Buffer, w: number, h: number) {
  return { warpedBuffer: buffer, confidence: 0.9, method: 'direct_upload' as const };
}

// ─── Pipeline principal ───────────────────────────────────────────────────────

export async function runBiometricPipeline(
  input: BiometricPipelineInput,
): Promise<BiometricPipelineResult> {

  const startTime = Date.now();
  console.log('[Pipeline] ═══════════════════════════════════════════');
  console.log('[Pipeline] Démarrage pipeline biométrique');

  try {
    // ── Étape 1 : Normaliser images ───────────────────────────────────────────
    const { buffer: docBuffer, width: docW, height: docH } =
      await normalizeInput(input.documentImageBase64);
    const { buffer: selfieBuffer, width: selfieW, height: selfieH } =
      await normalizeInput(input.selfieBase64);

    console.log(`[Pipeline] Doc: ${docW}×${docH} | Selfie: ${selfieW}×${selfieH}`);

    // ── Étape 2 : Détecter document ───────────────────────────────────────────
    const docDetection = await detectDocument(docBuffer, docW, docH);
    console.log(`[Pipeline] Doc: confiance=${docDetection.confidence.toFixed(2)} méthode=${docDetection.method}`);

    // ── Étape 3 : Localiser portrait sur le document ──────────────────────────
    const portraitZone = await locatePortrait(docDetection.warpedBuffer, docW, docH);
    console.log(`[Pipeline] Portrait doc: méthode=${portraitZone.method} confiance=${portraitZone.confidence.toFixed(2)}`);
    console.log(`[Pipeline] Portrait zone: left=${portraitZone.left} top=${portraitZone.top} w=${portraitZone.width} h=${portraitZone.height}`);

    if (portraitZone.confidence < 0.25) {
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: 'Portrait non localisé sur le document',
        debug: {
          docDetectionConfidence:      docDetection.confidence,
          portraitDetectionMethod:     portraitZone.method,
          portraitDetectionConfidence: portraitZone.confidence,
          selfieDetectionMethod:       'none',
          selfieDetectionConfidence:   0,
          embeddingMethod:             'none',
          durationMs:                  Date.now() - startTime,
        },
      };
    }

    // ── Étape 4 : Localiser visage sur le selfie ──────────────────────────────
    const selfieZone = await locatePortrait(selfieBuffer, selfieW, selfieH);
    console.log(`[Pipeline] Portrait selfie: méthode=${selfieZone.method} confiance=${selfieZone.confidence.toFixed(2)}`);

    // Si pas de visage détecté sur le selfie → utiliser toute l'image
    const selfieExtract = selfieZone.confidence >= 0.4
      ? selfieZone
      : { left: 0, top: 0, width: selfieW, height: selfieH };

    if (selfieZone.confidence < 0.4) {
      console.log('[Pipeline] Selfie: pas de visage détecté, utilisation image complète');
    }

    // ── Étape 5 : Crop 112×112 pour ArcFace ──────────────────────────────────
    const [docFace, selfieFace] = await Promise.all([
      cropAndResize(docDetection.warpedBuffer, portraitZone),
      cropAndResize(selfieBuffer, selfieExtract),
    ]);

    console.log(`[Pipeline] Faces extraites — doc: ${docFace.length}B selfie: ${selfieFace.length}B`);

    // ── Étape 6 : Générer embeddings ArcFace ─────────────────────────────────
    const [docEmbed, selfieEmbed] = await Promise.all([
      generateEmbedding(docFace),
      generateEmbedding(selfieFace),
    ]);

    const embeddingMethod = docEmbed.method ?? 'unknown';
    console.log(`[Pipeline] Embeddings: méthode=${embeddingMethod} doc_conf=${docEmbed.confidence.toFixed(2)} selfie_conf=${selfieEmbed.confidence.toFixed(2)}`);

    if (!docEmbed.embedding || !selfieEmbed.embedding) {
      const reason = docEmbed.error ?? selfieEmbed.error ?? 'Embedding échoué';
      console.error('[Pipeline] Embedding échoué:', reason);
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: reason,
        debug: {
          docDetectionConfidence:      docDetection.confidence,
          portraitDetectionMethod:     portraitZone.method,
          portraitDetectionConfidence: portraitZone.confidence,
          selfieDetectionMethod:       selfieZone.method,
          selfieDetectionConfidence:   selfieZone.confidence,
          embeddingMethod,
          durationMs:                  Date.now() - startTime,
        },
      };
    }

    // ── Étape 7 : Matching ────────────────────────────────────────────────────
    const matchResult = matchFaces(docEmbed.embedding, selfieEmbed.embedding, embeddingMethod);
    console.log(`[Pipeline] Match: score=${matchResult.similarityScore.toFixed(3)} matched=${matchResult.matched} risk=${matchResult.riskLevel}`);

    const elapsed = Date.now() - startTime;
    console.log(`[Pipeline] ✓ Terminé en ${elapsed}ms`);

    return {
      matched:           matchResult.matched,
      similarityScore:   matchResult.similarityScore,
      riskLevel:         matchResult.riskLevel,
      livenessConfirmed: false,
      needsManualReview: matchResult.needsManualReview,
      failureReason:     matchResult.failureReason,
      debug: {
        docDetectionConfidence:      docDetection.confidence,
        portraitDetectionMethod:     portraitZone.method,
        portraitDetectionConfidence: portraitZone.confidence,
        selfieDetectionMethod:       selfieZone.method,
        selfieDetectionConfidence:   selfieZone.confidence,
        embeddingMethod,
        durationMs:                  elapsed,
      },
    };

  } catch (err) {
    console.error('[Pipeline] Erreur fatale:', err);
    return {
      matched: false, similarityScore: 0, riskLevel: 'high',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: 'Erreur interne pipeline biométrique',
    };
  }
}