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
    embeddingMethod:             string;
    durationMs:                  number;
  };
}

// ─── Normalisation image ──────────────────────────────────────────────────────

async function normalizeInput(base64: string): Promise<{ buffer: Buffer; width: number; height: number }> {
  const data   = base64.replace(/^data:image\/\w+;base64,/, '');
  const buffer = Buffer.from(data, 'base64');
  const meta   = await sharp(buffer).metadata();
  const width  = meta.width  ?? 800;
  const height = meta.height ?? 600;

  // Redimensionner si trop grande (économie mémoire)
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

// ─── Amélioration portrait ────────────────────────────────────────────────────

async function enhancePortrait(
  imageBuffer: Buffer,
  zone: { left: number; top: number; width: number; height: number },
): Promise<Buffer> {
  return sharp(imageBuffer)
    .extract({ left: Math.round(zone.left), top: Math.round(zone.top), width: Math.round(zone.width), height: Math.round(zone.height) })
    .resize(224, 224, { fit: 'fill', kernel: 'lanczos3' })
    .normalize()
    .jpeg({ quality: 95 })
    .toBuffer();
}

// ─── Détection document (contours Canny simplifié) ────────────────────────────

async function detectDocument(buffer: Buffer, w: number, h: number) {
  // Sur Railway on a sharp — on fait une détection simple de confiance
  // La détection de coins Canny nécessite opencv4nodejs (trop lourd)
  // → on retourne le buffer tel quel avec confiance 0.9 (image uploadée manuellement = bonne qualité)
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
    // Étape 1 — Normaliser images
    const { buffer: docBuffer, width: docW, height: docH } =
      await normalizeInput(input.documentImageBase64);
    const { buffer: selfieBuffer, width: selfieW, height: selfieH } =
      await normalizeInput(input.selfieBase64);

    // Étape 2 — Détecter document
    const docDetection = await detectDocument(docBuffer, docW, docH);
    console.log(`[Pipeline] Doc: confiance=${docDetection.confidence.toFixed(2)} méthode=${docDetection.method}`);

    // Étape 3 — Localiser portrait sur le document
    const portraitZone = await locatePortrait(docDetection.warpedBuffer, docW, docH);
    console.log(`[Pipeline] Portrait: méthode=${portraitZone.method} confiance=${portraitZone.confidence.toFixed(2)}`);

    if (portraitZone.confidence < 0.25) {
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: 'Portrait non localisé sur le document',
        debug: {
          docDetectionConfidence:      docDetection.confidence,
          portraitDetectionMethod:     portraitZone.method,
          portraitDetectionConfidence: portraitZone.confidence,
          embeddingMethod:             'none',
          durationMs:                  Date.now() - startTime,
        },
      };
    }

    // Étape 4 — Extraire et améliorer portraits
    const enhancedPortrait = await enhancePortrait(docDetection.warpedBuffer, portraitZone);
    const enhancedSelfie   = await enhancePortrait(selfieBuffer, {
      left: 0, top: 0, width: selfieW, height: selfieH,
    });

    // Étape 5 — Générer embeddings
    const [docEmbed, selfieEmbed] = await Promise.all([
      generateEmbedding(enhancedPortrait),
      generateEmbedding(enhancedSelfie),
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
          embeddingMethod,
          durationMs:                  Date.now() - startTime,
        },
      };
    }

    // Étape 6 — Matching
    const matchResult = matchFaces(docEmbed.embedding, selfieEmbed.embedding, embeddingMethod);
    console.log(`[Pipeline] Match: score=${matchResult.similarityScore.toFixed(3)} matched=${matchResult.matched} risk=${matchResult.riskLevel}`);

    const elapsed = Date.now() - startTime;
    console.log(`[Pipeline] ✓ Terminé en ${elapsed}ms`);

    return {
      matched:           matchResult.matched,
      similarityScore:   matchResult.similarityScore,
      riskLevel:         matchResult.riskLevel,
      livenessConfirmed: false, // liveness non implémenté ici (côté client)
      needsManualReview: matchResult.needsManualReview,
      failureReason:     matchResult.failureReason,
      debug: {
        docDetectionConfidence:      docDetection.confidence,
        portraitDetectionMethod:     portraitZone.method,
        portraitDetectionConfidence: portraitZone.confidence,
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
