import type { FaceEmbedding } from './face-embedder';

export interface FaceMatchDecision {
  matched:           boolean;
  similarityScore:   number;
  riskLevel:         'low' | 'medium' | 'high';
  livenessConfirmed: boolean;
  needsManualReview: boolean;
  failureReason?:    string;
}

// ─── Seuils ArcFace 512-dim — document imprimé vs selfie ─────────────────────
//
// Contexte réel observé (carte CEDEAO portrait N&B vs selfie couleur) :
//   même personne   : score 0.62 – 0.72  (cosinus brut 0.24 – 0.44)
//   personne diffs  : score < 0.55       (cosinus brut < 0.10)
//
// Les seuils InsightFace "selfie vs selfie" (MATCH=0.64) sont trop stricts
// pour document imprimé vs selfie car :
//   - portrait carte = N&B ou sépia, basse résolution, parfois flou
//   - selfie = couleur, haute résolution, éclairage différent
//   - la distance de domaine abaisse mécaniquement le cosinus de ~0.08–0.12
//
// Calibration ajustée pour document vs selfie :
//   MATCH      : 0.60  — seuil minimal "même personne" (au lieu de 0.64)
//   LOW_RISK   : 0.68  — match fiable sans révision    (au lieu de 0.72)
//   MEDIUM_RISK: 0.52  — zone grise → révision manuelle (au lieu de 0.60)
//   REJECT     : <0.52 — visages différents

const THRESHOLDS_ONNX = {
  MATCH:       0.60,   // cosinus brut ≈ 0.20 — même personne doc vs selfie
  LOW_RISK:    0.68,   // cosinus brut ≈ 0.36 — match fiable
  MEDIUM_RISK: 0.52,   // cosinus brut ≈ 0.04 — zone grise
} as const;

const THRESHOLDS_CLAUDE = {
  MATCH:       0.82,
  LOW_RISK:    0.87,
  MEDIUM_RISK: 0.60,
} as const;

// ─── Similarité cosinus ───────────────────────────────────────────────────────
// Les embeddings ArcFace sont L2-normalisés → dot product = cosinus brut ∈ [-1, 1]
// On remet en [0, 1] via (dot + 1) / 2

export function cosineSimilarity(a: FaceEmbedding, b: FaceEmbedding): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return (Math.max(-1, Math.min(1, dot)) + 1) / 2;
}

// ─── Décision de matching ─────────────────────────────────────────────────────

export function matchFaces(
  embeddingDoc:    FaceEmbedding | null,
  embeddingSelfie: FaceEmbedding | null,
  method?: string,
): FaceMatchDecision {

  if (!embeddingDoc || !embeddingSelfie) {
    return {
      matched: false, similarityScore: 0, riskLevel: 'high',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: 'Embedding non généré — visage non détecté',
    };
  }

  const docNorm    = Math.sqrt(embeddingDoc.reduce((s, v) => s + v * v, 0));
  const selfieNorm = Math.sqrt(embeddingSelfie.reduce((s, v) => s + v * v, 0));
  if (docNorm < 0.1 || selfieNorm < 0.1) {
    return {
      matched: false, similarityScore: 0, riskLevel: 'high',
      livenessConfirmed: false, needsManualReview: false,
      failureReason: 'Embedding invalide — image suspecte',
    };
  }

  const similarity = cosineSimilarity(embeddingDoc, embeddingSelfie);
  const isClaude   = method === 'claude' || (embeddingDoc.length !== 512);
  const T          = isClaude ? THRESHOLDS_CLAUDE : THRESHOLDS_ONNX;

  const dotBrut = similarity * 2 - 1;
  console.log(
    `[Matcher] Similarité: ${similarity.toFixed(3)} (cosinus brut: ${dotBrut.toFixed(3)})` +
    ` | méthode: ${isClaude ? `claude-${embeddingDoc.length}dim` : 'onnx-512dim'}` +
    ` | seuils: MATCH=${T.MATCH} LOW=${T.LOW_RISK} MED=${T.MEDIUM_RISK}`
  );

  // ── Même personne — match fiable ─────────────────────────────────────────
  if (similarity >= T.LOW_RISK) {
    return {
      matched: true, similarityScore: similarity, riskLevel: 'low',
      livenessConfirmed: false, needsManualReview: false,
    };
  }

  // ── Probablement la même personne (qualité doc faible) ───────────────────
  if (similarity >= T.MATCH) {
    return {
      matched: true, similarityScore: similarity, riskLevel: 'medium',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: `Match probable mais score faible (${similarity.toFixed(3)}) — vérification recommandée`,
    };
  }

  // ── Zone grise — révision manuelle ───────────────────────────────────────
  if (similarity >= T.MEDIUM_RISK) {
    return {
      matched: false, similarityScore: similarity, riskLevel: 'medium',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: `Score intermédiaire (${similarity.toFixed(3)}) — révision manuelle`,
    };
  }

  // ── Visages différents ────────────────────────────────────────────────────
  return {
    matched: false, similarityScore: similarity, riskLevel: 'high',
    livenessConfirmed: false, needsManualReview: false,
    failureReason: `Visages différents (score: ${similarity.toFixed(3)}, cosinus: ${dotBrut.toFixed(3)})`,
  };
}