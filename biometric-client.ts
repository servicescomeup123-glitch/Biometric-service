// lib/biometric-client.ts
// Remplace l'import direct de runBiometricPipeline dans ton code Vercel
// par un appel HTTP vers le service Railway.

export interface BiometricPipelineResult {
  matched:           boolean;
  similarityScore:   number;
  riskLevel:         'low' | 'medium' | 'high';
  livenessConfirmed: boolean;
  needsManualReview: boolean;
  failureReason?:    string;
  debug?: Record<string, unknown>;
}

const BIOMETRIC_SERVICE_URL = process.env.BIOMETRIC_SERVICE_URL;
const SERVICE_SECRET_KEY    = process.env.SERVICE_SECRET_KEY;

export async function runBiometricPipeline(input: {
  documentImageBase64: string;
  selfieBase64: string;
}): Promise<BiometricPipelineResult> {

  if (!BIOMETRIC_SERVICE_URL) {
    console.error('[BiometricClient] BIOMETRIC_SERVICE_URL non configurée');
    return {
      matched: false, similarityScore: 0, riskLevel: 'high',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: 'Service biométrique non configuré',
    };
  }

  try {
    const response = await fetch(`${BIOMETRIC_SERVICE_URL}/biometric/match`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(SERVICE_SECRET_KEY ? { 'x-service-key': SERVICE_SECRET_KEY } : {}),
      },
      body: JSON.stringify(input),
      signal: AbortSignal.timeout(30000), // 30s max
    });

    if (!response.ok) {
      const text = await response.text();
      console.error('[BiometricClient] Erreur HTTP:', response.status, text);
      return {
        matched: false, similarityScore: 0, riskLevel: 'high',
        livenessConfirmed: false, needsManualReview: true,
        failureReason: `Erreur service biométrique: HTTP ${response.status}`,
      };
    }

    return await response.json() as BiometricPipelineResult;

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[BiometricClient] Erreur réseau:', msg);
    return {
      matched: false, similarityScore: 0, riskLevel: 'high',
      livenessConfirmed: false, needsManualReview: true,
      failureReason: `Timeout ou erreur réseau: ${msg}`,
    };
  }
}
