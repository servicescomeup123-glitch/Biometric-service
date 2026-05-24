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

// ─── Correction d'orientation du document ─────────────────────────────────────
// Une carte d'identité est toujours en paysage (largeur > hauteur).
// Si l'image est en portrait, on teste les deux rotations possibles (90° et -90°)
// et on garde celle qui donne le meilleur score de détection de visage.

async function correctDocumentOrientation(
  buffer: Buffer,
  width: number,
  height: number,
): Promise<{ buffer: Buffer; width: number; height: number }> {

  // Déjà en paysage → rien à faire
  if (width >= height) {
    return { buffer, width, height };
  }

  console.log(`[Pipeline] Document en portrait (${width}×${height}) — correction orientation...`);

  // Tenter rotation +90°
  const rot90 = await sharp(buffer).rotate(90).jpeg({ quality: 90 }).toBuffer();
  const meta90 = await sharp(rot90).metadata();
  const w90 = meta90.width ?? height;
  const h90 = meta90.height ?? width;

  // Tenter rotation -90°
  const rotN90 = await sharp(buffer).rotate(-90).jpeg({ quality: 90 }).toBuffer();
  const metaN90 = await sharp(rotN90).metadata();
  const wN90 = metaN90.width ?? height;
  const hN90 = metaN90.height ?? width;

  // Tester la détection de portrait sur les deux rotations
  // et garder celle avec la meilleure confiance
  const [zone90, zoneN90] = await Promise.all([
    locatePortrait(rot90,  w90,  h90),
    locatePortrait(rotN90, wN90, hN90),
  ]);

  console.log(
    `[Pipeline] Rotation +90°: confiance=${zone90.confidence.toFixed(2)}` +
    ` | Rotation -90°: confiance=${zoneN90.confidence.toFixed(2)}`
  );

  if (zone90.confidence >= zoneN90.confidence) {
    console.log('[Pipeline] → Rotation +90° retenue');
    return { buffer: rot90, width: w90, height: h90 };
  } else {
    console.log('[Pipeline] → Rotation -90° retenue');
    return { buffer: rotN90, width: wN90, height: hN90 };
  }
}

// ─── Crop + resize vers 112×112 pour ArcFace ─────────────────────────────────

async function cropAndResize(
  imageBuffer: Buffer,
  zone: { left: number; top: number; width: number; height: number },
  imageW: number,
  imageH: number,
): Promise<Buffer> {
  // Clamp explicite pour éviter les dépassements de bords
  const left   = Math.max(0, Math.round(zone.left));
  const top    = Math.max(0, Math.round(zone.top));
  const width  = Math.min(imageW - left, Math.max(1, Math.round(zone.width)));
  const height = Math.min(imageH - top,  Math.max(1, Math.round(zone.height)));

  return sharp(imageBuffer)
    .extract({ left, top, width, height })
    .resize(112, 112, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .jpeg({ quality: 95 })
    .toBuffer();
}

// ─── Validation du portrait extrait ──────────────────────────────────────────
// Un portrait trop petit (< 5% de la surface du document) est suspect —
// probablement une fausse détection sur un logo ou un tampon.

function isPortraitZoneValid(
  zone: { width: number; height: number; confidence: number },
  imageW: number,
  imageH: number,
): boolean {
  const docArea      = imageW * imageH;
  const portraitArea = zone.width * zone.height;
  const ratio        = portraitArea / docArea;

  if (ratio < 0.02) {
    console.warn(
      `[Pipeline] Portrait trop petit: ${zone.width}×${zone.height}` +
      ` = ${(ratio * 100).toFixed(1)}% de l'image — rejeté`
    );
    return false;
  }

  return true;
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
    const { buffer: docBufferRaw, width: docWRaw, height: docHRaw } =
      await normalizeInput(input.documentImageBase64);
    const { buffer: selfieBuffer, width: selfieW, height: selfieH } =
      await normalizeInput(input.selfieBase64);

    console.log(`[Pipeline] Doc: ${docWRaw}×${docHRaw} | Selfie: ${selfieW}×${selfieH}`);

    // ── Étape 2 : Corriger orientation du document ────────────────────────────
    // Les cartes d'identité sont en paysage. Si le document est en portrait,
    // on cherche la rotation qui place le visage correctement.
    const { buffer: docBuffer, width: docW, height: docH } =
      await correctDocumentOrientation(docBufferRaw, docWRaw, docHRaw);

    if (docW !== docWRaw || docH !== docHRaw) {
      console.log(`[Pipeline] Doc après correction: ${docW}×${docH}`);
    }

    // ── Étape 3 : Détecter document ───────────────────────────────────────────
    const docDetection = await detectDocument(docBuffer, docW, docH);
    console.log(`[Pipeline] Doc: confiance=${docDetection.confidence.toFixed(2)} méthode=${docDetection.method}`);

    // ── Étape 4 : Localiser portrait sur le document ──────────────────────────
    const portraitZone = await locatePortrait(docDetection.warpedBuffer, docW, docH);
    console.log(`[Pipeline] Portrait doc: méthode=${portraitZone.method} confiance=${portraitZone.confidence.toFixed(2)}`);
    console.log(`[Pipeline] Portrait zone: left=${portraitZone.left} top=${portraitZone.top} w=${portraitZone.width} h=${portraitZone.height}`);

    // Rejeter si confiance trop faible OU zone trop petite (fausse détection)
    if (portraitZone.confidence < 0.25 || !isPortraitZoneValid(portraitZone, docW, docH)) {
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: 'Portrait non localisé sur le document — veuillez reprendre la photo du document',
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

    // ── Étape 5 : Localiser visage sur le selfie ──────────────────────────────
    const selfieZone = await locatePortrait(selfieBuffer, selfieW, selfieH);
    console.log(`[Pipeline] Portrait selfie: méthode=${selfieZone.method} confiance=${selfieZone.confidence.toFixed(2)}`);

    // Pour le selfie : si pas de visage détecté, utiliser toute l'image
    // (le selfie est une photo de face, ArcFace s'en sort mieux que sur un doc entier)
    const selfieExtract = (selfieZone.confidence >= 0.4 && isPortraitZoneValid(selfieZone, selfieW, selfieH))
      ? selfieZone
      : { left: 0, top: 0, width: selfieW, height: selfieH };

    if (selfieZone.confidence < 0.4) {
      console.log('[Pipeline] Selfie: pas de visage détecté, utilisation image complète');
    }

    // ── Étape 6 : Crop 112×112 pour ArcFace ──────────────────────────────────
    const [docFace, selfieFace] = await Promise.all([
      cropAndResize(docDetection.warpedBuffer, portraitZone, docW, docH),
      cropAndResize(selfieBuffer, selfieExtract, selfieW, selfieH),
    ]);

    console.log(`[Pipeline] Faces extraites — doc: ${docFace.length}B selfie: ${selfieFace.length}B`);

    // ── Étape 7 : Générer embeddings ArcFace ─────────────────────────────────
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

    // ── Étape 8 : Matching ────────────────────────────────────────────────────
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