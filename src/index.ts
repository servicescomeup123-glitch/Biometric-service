import express, { Request, Response, NextFunction } from 'express';
import { runBiometricPipeline } from './pipeline';
import * as fs from 'fs';

const app  = express();
const PORT = process.env.PORT || 3001;

let modelsReady = false;

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

function readyMiddleware(_req: Request, res: Response, next: NextFunction) {
  if (!modelsReady) {
    return res.status(503).json({ error: 'Service en cours d\'initialisation, réessayez dans quelques secondes.' });
  }
  next();
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status:      modelsReady ? 'ok' : 'starting',
    modelsReady,
    uptime:      Math.floor(process.uptime()),
  });
});

app.post('/biometric/match', authMiddleware, readyMiddleware, async (req: Request, res: Response) => {
  const { documentImageBase64, selfieBase64 } = req.body;

  if (!documentImageBase64 || !selfieBase64) {
    return res.status(400).json({ error: 'documentImageBase64 et selfieBase64 sont requis' });
  }
  if (typeof documentImageBase64 !== 'string' || typeof selfieBase64 !== 'string') {
    return res.status(400).json({ error: 'Les images doivent être des strings base64' });
  }

  console.log(
    `[API] POST /biometric/match — doc: ${Math.round(documentImageBase64.length / 1024)}KB` +
    ` selfie: ${Math.round(selfieBase64.length / 1024)}KB`,
  );

  // ── Sauvegarde des images pour diagnostic ─────────────────────────────────
  try {
    const docData    = documentImageBase64.replace(/^data:image\/\w+;base64,/, '');
    const selfieData = selfieBase64.replace(/^data:image\/\w+;base64,/, '');
    fs.writeFileSync('/tmp/last_doc.jpg',    Buffer.from(docData,    'base64'));
    fs.writeFileSync('/tmp/last_selfie.jpg', Buffer.from(selfieData, 'base64'));
    console.log('[Debug] Images sauvegardées dans /tmp/');
  } catch (e) {
    console.warn('[Debug] Impossible de sauvegarder les images:', e);
  }

  const result = await runBiometricPipeline({ documentImageBase64, selfieBase64 });
  return res.json(result);
});

// ─── Endpoints de debug (à retirer en production) ─────────────────────────────

app.get('/debug/doc', (_req, res) => {
  if (fs.existsSync('/tmp/last_doc.jpg')) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.send(fs.readFileSync('/tmp/last_doc.jpg'));
  } else {
    res.status(404).send('Pas encore de document reçu');
  }
});

app.get('/debug/selfie', (_req, res) => {
  if (fs.existsSync('/tmp/last_selfie.jpg')) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.send(fs.readFileSync('/tmp/last_selfie.jpg'));
  } else {
    res.status(404).send('Pas encore de selfie reçu');
  }
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

    if (arcfaceSession) {
      console.log('[Server] ✓ ArcFace   — inputs:', JSON.stringify(arcfaceSession.inputNames), '| outputs:', JSON.stringify(arcfaceSession.outputNames));
    } else {
      console.error('[Server] ✗ ArcFace ÉCHEC — fallback Claude Vision actif');
    }

    if (detectorSession) {
      console.log('[Server] ✓ Détecteur — inputs:', JSON.stringify(detectorSession.inputNames), '| outputs:', JSON.stringify(detectorSession.outputNames));
    } else {
      console.error('[Server] ✗ Détecteur ÉCHEC — fallback heuristique actif');
    }

  } catch (err) {
    console.error('[Server] Erreur pré-chargement modèles:', err);
  }

  modelsReady = true;
  console.log('[Server] ✓ Modèles prêts — service opérationnel.');
});