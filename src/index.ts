import express, { Request, Response, NextFunction } from 'express';
import { runBiometricPipeline } from './pipeline';

const app  = express();
const PORT = process.env.PORT || 3001;

// ─── Middleware ───────────────────────────────────────────────────────────────

app.use(express.json({ limit: '20mb' }));

function authMiddleware(req: Request, res: Response, next: NextFunction) {
  const key = req.headers['x-service-key'];
  if (!process.env.SERVICE_SECRET_KEY) return next();
  if (key !== process.env.SERVICE_SECRET_KEY) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', uptime: Math.floor(process.uptime()) });
});

app.post('/biometric/match', authMiddleware, async (req: Request, res: Response) => {
  const { documentImageBase64, selfieBase64 } = req.body;

  if (!documentImageBase64 || !selfieBase64) {
    return res.status(400).json({
      error: 'documentImageBase64 et selfieBase64 sont requis',
    });
  }

  if (typeof documentImageBase64 !== 'string' || typeof selfieBase64 !== 'string') {
    return res.status(400).json({ error: 'Les images doivent être des strings base64' });
  }

  console.log(
    `[API] POST /biometric/match — doc: ${Math.round(documentImageBase64.length / 1024)}KB` +
    ` selfie: ${Math.round(selfieBase64.length / 1024)}KB`,
  );

  const result = await runBiometricPipeline({ documentImageBase64, selfieBase64 });
  return res.json(result);
});

// ─── Démarrage ────────────────────────────────────────────────────────────────

app.listen(PORT, async () => {
  console.log(`[Server] Biometric service démarré sur le port ${PORT}`);
  console.log(`[Server] SERVICE_SECRET_KEY: ${process.env.SERVICE_SECRET_KEY ? '✓ configurée' : '⚠ non configurée (dev mode)'}`);
  console.log(`[Server] R2_MODELS_BASE_URL: ${process.env.R2_MODELS_BASE_URL ?? 'défaut pub-91a...'}`);
  console.log(`[Server] ANTHROPIC_API_KEY:  ${process.env.ANTHROPIC_API_KEY  ? '✓ configurée' : '✗ manquante'}`);

  console.log('[Server] Pré-chargement des modèles ONNX...');

  try {
    const { getArcFaceSession } = await import('./face-embedder');
    const { getDetector }       = await import('./portrait-locator');

    const [arcfaceSession, detectorSession] = await Promise.all([
      getArcFaceSession(),
      getDetector(),
    ]);

    // ── ArcFace ──────────────────────────────────────────────────────────────
    if (arcfaceSession) {
      console.log('[Server] ✓ ArcFace ONNX chargé');
      console.log('[Server]   ArcFace inputs :', JSON.stringify(arcfaceSession.inputNames));
      console.log('[Server]   ArcFace outputs:', JSON.stringify(arcfaceSession.outputNames));
    } else {
      console.error('[Server] ✗ ArcFace ONNX ÉCHEC — fallback Claude Vision actif');
    }

    // ── Détecteur portrait ───────────────────────────────────────────────────
    if (detectorSession) {
      console.log('[Server] ✓ Détecteur portrait ONNX chargé');
      console.log('[Server]   Detector inputs :', JSON.stringify(detectorSession.inputNames));
      console.log('[Server]   Detector outputs:', JSON.stringify(detectorSession.outputNames));

      // Log les dims de chaque output pour diagnostiquer le parsing SCRFD
      // On fait un run de test avec une image noire 640×640
      try {
        const ort         = require('onnxruntime-node');
        const dummyTensor = new ort.Tensor('float32', new Float32Array(3 * 640 * 640), [1, 3, 640, 640]);
        const inputName   = detectorSession.inputNames[0];
        const dummyResult = await detectorSession.run({ [inputName]: dummyTensor });

        console.log('[Server]   Detector output sizes (run test):');
        for (const name of detectorSession.outputNames) {
          const data = dummyResult[name]?.data as Float32Array | undefined;
          const dims = dummyResult[name]?.dims as number[] | undefined;
          console.log(`[Server]     "${name}" → length=${data?.length ?? 'N/A'} dims=${JSON.stringify(dims ?? [])}`);
        }
      } catch (testErr) {
        console.warn('[Server]   Run test détecteur échoué:', testErr);
      }

    } else {
      console.error('[Server] ✗ Détecteur portrait ONNX ÉCHEC — fallback heuristique actif');
    }

  } catch (err) {
    console.error('[Server] Erreur pré-chargement modèles:', err);
  }

  console.log('[Server] Modèles prêts — service opérationnel.');
});