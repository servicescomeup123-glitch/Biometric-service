import express, { Request, Response, NextFunction } from 'express';
import { runBiometricPipeline } from './pipeline';

const app  = express();
const PORT = process.env.PORT || 3001;

// ─── Middleware ───────────────────────────────────────────────────────────────

app.use(express.json({ limit: '20mb' })); // images base64 peuvent être lourdes

// Auth par clé secrète partagée avec Vercel
function authMiddleware(req: Request, res: Response, next: NextFunction) {
  const key = req.headers['x-service-key'];
  if (!process.env.SERVICE_SECRET_KEY) {
    // Pas de clé configurée → accepter en dev
    return next();
  }
  if (key !== process.env.SERVICE_SECRET_KEY) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health check — Railway ping ce endpoint pour savoir si le service est up
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', uptime: Math.floor(process.uptime()) });
});

// Endpoint principal — appelé depuis ton API Vercel
app.post('/biometric/match', authMiddleware, async (req: Request, res: Response) => {
  const { documentImageBase64, selfieBase64 } = req.body;

  if (!documentImageBase64 || !selfieBase64) {
    return res.status(400).json({
      error: 'documentImageBase64 et selfieBase64 sont requis',
    });
  }

  // Vérifications basiques de format
  if (typeof documentImageBase64 !== 'string' || typeof selfieBase64 !== 'string') {
    return res.status(400).json({ error: 'Les images doivent être des strings base64' });
  }

  console.log(`[API] POST /biometric/match — doc: ${Math.round(documentImageBase64.length / 1024)}KB selfie: ${Math.round(selfieBase64.length / 1024)}KB`);

  const result = await runBiometricPipeline({ documentImageBase64, selfieBase64 });

  return res.json(result);
});

// ─── Démarrage ────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[Server] Biometric service démarré sur le port ${PORT}`);
  console.log(`[Server] SERVICE_SECRET_KEY: ${process.env.SERVICE_SECRET_KEY ? '✓ configurée' : '⚠ non configurée (dev mode)'}`);
  console.log(`[Server] R2_MODELS_BASE_URL: ${process.env.R2_MODELS_BASE_URL ?? 'défaut pub-91a...'}`);
  console.log(`[Server] ANTHROPIC_API_KEY:  ${process.env.ANTHROPIC_API_KEY ? '✓ configurée' : '✗ manquante'}`);
});
