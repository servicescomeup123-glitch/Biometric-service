import type { FaceEmbedding } from './face-embedder';

export interface FaceMatchDecision {
  matched:           boolean;
  similarityScore:   number;
  riskLevel:         'low' | 'medium' | 'high';
  livenessConfirmed: boolean;
  needsManualReview: boolean;
  failureReason?:    string;
}

// ─── Seuils adaptatifs ────────────────────────────────────────────────────────

const THRESHOLDS_ONNX = {
  LOW_RISK:    0.60,
  MEDIUM_RISK: 0.40,
} as const;

const THRESHOLDS_CLAUDE = {
  LOW_RISK:    0.82,
  MEDIUM_RISK: 0.60,
} as const;

// ─── Similarité cosinus ───────────────────────────────────────────────────────

export function cosineSimilarity(a: FaceEmbedding, b: FaceEmbedding): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return (dot + 1) / 2;
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
  const isClaude   = method === 'claude' || embeddingDoc.length === 16;
  const T          = isClaude ? THRESHOLDS_CLAUDE : THRESHOLDS_ONNX;

  console.log(`[Matcher] Similarité cosinus: ${similarity.toFixed(3)} | méthode: ${isClaude ? 'claude-16dim' : 'onnx-512dim'} | seuils: LOW=${T.LOW_RISK} MED=${T.MEDIUM_RISK}`);

  if (similarity >= T.LOW_RISK) {
    return { matched: true, similarityScore: similarity, riskLevel: 'low', livenessConfirmed: false, needsManualReview: false };
  }

  if (similarity >= T.MEDIUM_RISK) {
    return {
      matched: false, similarityScore: similarity, riskLevel: 'medium',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: `Score intermédiaire (${similarity.toFixed(3)}) — révision manuelle`,
    };
  }

  return {
    matched: false, similarityScore: similarity, riskLevel: 'high',
    livenessConfirmed: false, needsManualReview: false,
    failureReason: `Visages différents (score: ${similarity.toFixed(3)})`,
  };
}
