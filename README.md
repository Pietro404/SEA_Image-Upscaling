# Image Upscaling: Super-Resolution
Prima versione del repo di Sistemi di Elaborazione Accelerata (SEA)

Vicino Più Prossimo:

    Copia il valore del pixel più vicino nella nuova posizione;
    Produce immagini nitide ma con effetto a blocchi;
    Veloce ma priva di uniformità e dettaglio.

Bilineare:

    Calcola la media di quattro pixel adiacenti per stimare il nuovo valore del pixel;
    Produce immagini più uniformi ma può introdurre sfocatura.

Bicubica:

    Utilizza una media ponderata di 16 pixel circostanti;
    Garantisce maggiore uniformità e nitidezza rispetto all'interpolazione bilineare.
