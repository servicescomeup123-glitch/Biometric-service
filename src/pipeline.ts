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

// ─── Extraction du portrait via Claude Vision ─────────────────────────────────
// Claude reçoit l'image du document entier et retourne les coordonnées
// du portrait en pourcentage (0-100) pour être indépendant de la résolution.

async function extractPortraitWithClaude(
  imageBuffer: Buffer,
  W: number,
  H: number,
): Promise<{ left: number; top: number; width: number; height: number; confidence: number } | null> {

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.warn('[Claude] ANTHROPIC_API_KEY manquante — skip extraction Claude');
    return null;
  }

  try {
    const base64 = imageBuffer.toString('base64');

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type':      'application/json',
        'x-api-key':         apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model:      'claude-sonnet-4-5',
        max_tokens: 256,
        messages: [{
          role: 'user',
          content: [
            {
              type:   'image',
              source: { type: 'base64', media_type: 'image/jpeg', data: base64 },
            },
            {
              type: 'text',
              text: `Cette image contient un document d'identité (carte d'identité, passeport, permis).
Localise UNIQUEMENT le portrait photo (visage de la personne) sur ce document.
Réponds UNIQUEMENT en JSON strict, sans markdown :
{
  "found": true,
  "x_pct": 15,
  "y_pct": 30,
  "w_pct": 20,
  "h_pct": 35
}
Où x_pct/y_pct sont le coin supérieur gauche du portrait en % de l'image, et w_pct/h_pct la largeur/hauteur en %.
Si tu ne vois pas de portrait clairement : { "found": false }`,
            },
          ],
        }],
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      console.error('[Claude] Erreur API:', response.status);
      return null;
    }

    const result = await response.json();
    const text   = (result.content?.[0]?.text ?? '').trim();

    let parsed: any;
    try {
      parsed = JSON.parse(text.replace(/```json\s*/g, '').replace(/```/g, '').trim());
    } catch {
      const m = text.match(/\{[\s\S]*\}/);
      if (!m) return null;
      parsed = JSON.parse(m[0]);
    }

    if (!parsed.found) {
      console.log('[Claude] Portrait non trouvé dans le document');
      return null;
    }

    // Convertir les pourcentages en pixels
    const left   = Math.max(0, Math.round(parsed.x_pct / 100 * W));
    const top    = Math.max(0, Math.round(parsed.y_pct / 100 * H));
    const width  = Math.min(W - left, Math.round(parsed.w_pct / 100 * W));
    const height = Math.min(H - top,  Math.round(parsed.h_pct / 100 * H));

    console.log(`[Claude] Portrait trouvé: [${left},${top} ${width}×${height}]`);

    return { left, top, width, height, confidence: 0.85 };

  } catch (err) {
    console.error('[Claude] Erreur extraction portrait:', err);
    return null;
  }
}

// ─── Crop + resize vers 112×112 pour ArcFace ─────────────────────────────────

async function cropAndResize(
  imageBuffer: Buffer,
  zone: { left: number; top: number; width: number; height: number },
  imageW: number,
  imageH: number,
): Promise<Buffer> {
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

    // ── Étape 3 : Localiser portrait — Claude Vision en priorité ──────────────
    // ONNX échoue quand le portrait est petit (carte posée sur table).
    // Claude Vision comprend la scène et localise le portrait correctement.
    let portraitZone: { left: number; top: number; width: number; height: number; confidence: number; method: string } | null = null;

    // 3a. Essayer Claude Vision d'abord
    const claudeZone = await extractPortraitWithClaude(docDetection.warpedBuffer, docW, docH);
    if (claudeZone) {
      portraitZone = { ...claudeZone, method: 'claude_vision' };
      console.log(`[Pipeline] Portrait doc (Claude): méthode=claude_vision confiance=${portraitZone.confidence.toFixed(2)}`);
    }

    // 3b. Fallback ONNX si Claude échoue
    if (!portraitZone) {
      console.log('[Pipeline] Claude Vision indisponible — fallback ONNX');
      const onnxZone = await locatePortrait(docDetection.warpedBuffer, docW, docH);
      if (onnxZone.confidence >= 0.3) {
        portraitZone = onnxZone;
        console.log(`[Pipeline] Portrait doc (ONNX): méthode=${onnxZone.method} confiance=${onnxZone.confidence.toFixed(2)}`);
      }
    }

    // 3c. Blocage si aucun portrait trouvé
    if (!portraitZone || portraitZone.confidence < 0.2) {
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: 'Portrait non localisé sur le document — veuillez reprendre la photo du document en vous assurant que le recto de la carte est bien visible',
        debug: {
          docDetectionConfidence:      docDetection.confidence,
          portraitDetectionMethod:     portraitZone?.method ?? 'none',
          portraitDetectionConfidence: portraitZone?.confidence ?? 0,
          selfieDetectionMethod:       'none',
          selfieDetectionConfidence:   0,
          embeddingMethod:             'none',
          durationMs:                  Date.now() - startTime,
        },
      };
    }

    console.log(`[Pipeline] Portrait zone: left=${portraitZone.left} top=${portraitZone.top} w=${portraitZone.width} h=${portraitZone.height}`);

    // ── Étape 4 : Localiser visage sur le selfie ──────────────────────────────
    const selfieZone = await locatePortrait(selfieBuffer, selfieW, selfieH);
    console.log(`[Pipeline] Portrait selfie: méthode=${selfieZone.method} confiance=${selfieZone.confidence.toFixed(2)}`);

    const selfieExtract = selfieZone.confidence >= 0.3
      ? selfieZone
      : { left: 0, top: 0, width: selfieW, height: selfieH };

    if (selfieZone.confidence < 0.3) {
      console.log('[Pipeline] Selfie: pas de visage ONNX, utilisation image complète');
    }

    // ── Étape 5 : Crop 112×112 pour ArcFace ──────────────────────────────────
    const [docFace, selfieFace] = await Promise.all([
      cropAndResize(docDetection.warpedBuffer, portraitZone, docW, docH),
      cropAndResize(selfieBuffer, selfieExtract, selfieW, selfieH),
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