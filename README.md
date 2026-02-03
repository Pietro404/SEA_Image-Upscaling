# Image Upscaling: Super-Resolution
Repository Progetto Sistemi di Elaborazione Accelerata (SEA)             CITO, GIROTTI 2025/2026
Vicino Più Prossimo:

    Copia il valore del pixel più vicino nella nuova posizione;
    Produce immagini nitide ma con effetto a blocchi;
    La più veloce (1pixel analizzato) ma priva di uniformità e dettaglio.

Bilineare:

    Calcola la media di quattro pixel adiacenti per stimare il nuovo valore del pixel;
    Produce immagini più uniformi ma può introdurre sfocatura.
    Velocità media: più lenta della NN (4 pixel analizzati, 2x2)

Bicubica:

    Utilizza una media ponderata di 16 pixel circostanti (4x4);
    Garantisce maggiore uniformità e nitidezza rispetto all'interpolazione bilineare.
    Velocità ridotta: più lenta della bilineare
<img width="756" height="345" alt="image" src="https://github.com/user-attachments/assets/4f4f1204-93e6-4807-a8b8-ad92d13eefdb" />
