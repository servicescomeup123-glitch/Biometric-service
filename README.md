# Biometric Service — Railway

Microservice Express pour le pipeline biométrique ArcFace ONNX.
Tourne sur Railway (free tier avec cold start, ou $5/mois sans cold start).

## Pourquoi ce service ?

`onnxruntime-node` ne fonctionne pas sur Vercel Hobby (pas de binaires natifs).
Ce service tourne sur Railway qui supporte Node.js natif complet.

## Déploiement sur Railway

### 1. Créer le repo GitHub
```bash
git init
git add .
git commit -m "init biometric service"
gh repo create biometric-service --private --push
```

### 2. Créer le projet sur Railway
- Va sur https://railway.app
- New Project → Deploy from GitHub repo
- Sélectionne ton repo `biometric-service`
- Railway détecte le Dockerfile automatiquement

### 3. Configurer les variables d'environnement
Dans Railway → ton service → Variables :

```
SERVICE_SECRET_KEY=<openssl rand -hex 32>
ANTHROPIC_API_KEY=sk-ant-...
R2_MODELS_BASE_URL=https://pub-91a604b2df2f4a17b8aa07c2c2eee859.r2.dev/models/buffalo_l
```

### 4. Récupérer l'URL du service
Railway → Settings → Networking → Generate Domain
Tu obtiens : `https://biometric-service-production-xxxx.up.railway.app`

### 5. Configurer Vercel
Dans ton projet Vercel → Settings → Environment Variables :

```
BIOMETRIC_SERVICE_URL=https://biometric-service-production-xxxx.up.railway.app
SERVICE_SECRET_KEY=<même clé que Railway>
```

### 6. Mettre à jour ton code Vercel
Dans ton fichier qui appelle `runBiometricPipeline`, remplace l'import :

```ts
// Avant (import local qui crashe sur Vercel) :
import { runBiometricPipeline } from '@/lib/biometric-pipeline';

// Après (appel HTTP vers Railway) :
import { runBiometricPipeline } from '@/lib/biometric-client';
```

Le fichier `biometric-client.ts` est fourni dans ce repo.

## Endpoints

### GET /health
Vérification que le service est up.
```json
{ "status": "ok", "uptime": 42 }
```

### POST /biometric/match
Header requis : `x-service-key: <SERVICE_SECRET_KEY>`

Body :
```json
{
  "documentImageBase64": "data:image/jpeg;base64,...",
  "selfieBase64": "data:image/jpeg;base64,..."
}
```

Réponse :
```json
{
  "matched": true,
  "similarityScore": 0.923,
  "riskLevel": "low",
  "livenessConfirmed": false,
  "needsManualReview": false,
  "debug": {
    "portraitDetectionMethod": "onnx_face_detection",
    "embeddingMethod": "onnx",
    "durationMs": 1240
  }
}
```

## Architecture

```
Vercel (Next.js)
  └── /api/v1/kyc/verify
        └── biometric-client.ts  ──HTTP──►  Railway (ce service)
                                              └── pipeline.ts
                                                    ├── portrait-locator.ts (ONNX det_10g)
                                                    ├── face-embedder.ts    (ONNX ArcFace)
                                                    └── face-matcher.ts     (cosinus)
```

## Notes

- Le premier appel après un cold start (~5-10s) télécharge les modèles depuis R2.
  Les appels suivants utilisent le cache en mémoire (arcfaceSession singleton).
- Sur Railway free tier, le service s'endort après inactivité.
  Pour éviter ça, ajoute un cron qui ping /health toutes les 5 minutes.
