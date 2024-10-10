# Esame di Deep Learning con Applicazioni

## Physics-Guided Architecture of Neural Networks for Quantifying Uncertainty in Lake Temperature Modeling

Regressione dei profili giornalieri di temperatura e densità a diverse profondità del Falling Creek Reservoir a partire da un set di dati meteorologici. Si ripercorre il lavoro fatto da *Daw et al.* in [1], approfondendo alcuni punti.

La cartella [PGA_LSTM](PGA_LSTM) contiene il codice e i dati forniti dagli autori del paper nella loro [repository github](https://github.com/arkadaw9/PGA_LSTM). Alcuni file sono stati modificati con commenti per rendere più chiaro l'intento degli autori, ma tale codice non viene mai chiamato dalle mie procedure.

## Dati

### Predittori e target

Set di 3 variabili:

- profondità (in m): quota sotto la superfice a cui è stata effettuata la misura di temperatura;
- temperatura GLM (in °C): temperatura stimata tramite General Lake Model 3.0;
- temperatura osservata (in °C)

|    | date                |   depth |   glm_temp |    temp |
|---:|:--------------------|--------:|-----------:|--------:|
|  0 | 2013-05-16 |    0.1  |  0.0172942 | 18.3691 |
|  1 | 2013-05-16 |    0.33 |  1.19269   | 18.3691 |
|  2 | 2013-05-16 |    0.67 |  4.24186   | 17.3454 |
|  3 | 2013-05-16 |    1    |  4.55674   | 16.8125 |
|  4 | 2013-05-16 |    1.33 |  4.55674   | 15.6954 |

Si dispone dei dati rilevati tra il 2013-05-16 e il 2018-12-31, per quanto non siano continuativi.

Tabella completa in [PGA_LSTM/Datasets/FCR_2013_2018_Observed_with_GLM_output.csv](PGA_LSTM/Datasets/FCR_2013_2018_Observed_with_GLM_output.csv).

### Meteorologici

Set di 10 variabili meteorologiche rilevate a scala giornaliera. Si dispone del periodo dal 2013-05-15 al 2018-12-31.

|    | date                |   ShortWave |   LongWave |   AirTemp |   RelHum |   WindSpeed |     Rain |   InFlowRate |   InFlowTemp |   SSSFlowRate |   SSSFlowOxygen |
|---:|:--------------------|------------:|-----------:|----------:|---------:|------------:|---------:|-------------:|-------------:|--------------:|----------------:|
|  0 | 2013-05-15 |     291.42  |    341.075 |   19.5346 |  67.5217 |     4.89219 | 0        |    0.0160159 |      15.6645 |     0.0151333 |          597.51 |
|  1 | 2013-05-16 |     315.067 |    345.671 |   22.0804 |  66.6287 |     5.28623 | 0        |    0.0149698 |      15.3409 |     0.0151333 |          597.51 |
|  2 | 2013-05-17 |     265.867 |    370.547 |   21.9021 |  67.9848 |     2.47276 | 0.124066 |    0.0141288 |      15.2883 |     0.0151333 |          597.51 |
|  3 | 2013-05-18 |     237.72  |    388.751 |   22.1775 |  75.1037 |     3.14095 | 0.253997 |    0.0170094 |      14.9034 |     0.0151333 |          597.51 |
|  4 | 2013-05-19 |     224.536 |    385.807 |   20.7142 |  85.7515 |     2.99334 | 0.253406 |    0.0241258 |      14.6352 |     0.0151333 |          597.51 |

Tabella completa in [PGA_LSTM/Datasets/FCR_2013_2018_Drivers.csv](PGA_LSTM/Datasets/FCR_2013_2018_Drivers.csv)

## Codice
Codice e risultati intermedi sono ripartiti nelle seguenti cartelle:
- [data/](data/): embeddings prodotti da varie versioni dell'autoencoder;
- [notebooks/](notebooks/): notebook jupyter utilizzati per effettuare calcoli e data exploration:
  - [notebooks/autoencoder_pytorch.ipynb](notebooks/autoencoder_pytorch.ipynb) training ed evaluation dell'autoencoder;
  - [notebooks/presentazione/presentazione.ipynb](notebooks/presentazione/presentazione.ipynb) training, evaluation e output dei modelli presentati in [notebooks/presentazione/esame DLA.pptx](notebooks/presentazione/esame%20DLA.pptx);
- [notebooks/presentazione/](notebooks/presentazione/): asset usati nella presentazione;
- [PGA_LSTM/](PGA_LSTM/): codice e dati forniti dagli autori del paper;
- [results/](results/): checkpoint con i migliori modelli per ogni combinazione di parametro/seed random ricercata;
- [src/](src/): codice sorgente;
- [src/scripts](src/scripts/) script lanciabili da terminale per fare le ricerche dei parametri.

I moduli più significativi sono nella cartella [models](src/models/):
- [pga.py](src/models/pga.py): cella LSTM custom che produce sequenze monotone;
- [regressors.py](src/models/regressors.py) e [regressors_v2.py](src/models/regressors_v2.py): reti composite utilizzate come regressori; i moduli "_v2" sono pensati per essere utilizzati secondo il mio approccio di integrazione dell'embedding meteorologico in una procedura unica (invece che autoencoder separato da regressor).

Il resto del codice sono procedure per caricare i dati ([src/datasets](src/datasets/)) e boilerplate di vario genere.

Le classi il cui nome inizia con "Their" sono state scritte cercando di riprodurre alla perfezione le scelte degli autori. In alcuni casi questo si traduce in *educated guesses*, dal momento che non le informazioni fornite non sono complete.

### Caveat

Le librerie necessarie ad eseguire il codice dovrebbero essere tutte listate in [environment.yml](environment.yml).

Il codice non è organizzato in maniera molto efficace né commentato a dovere. Alcune sezioni potrebbero non funzionare perché non aggiornate.

Le procedure di ricerca dei parametri si aspettano di registrare i risultati su un database Postgres. È possibile cambiare il tipo di storage seguendo la [documentazione di optuna](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).



[1] *Daw, A., Thomas, R. Q., Carey, C. C., Read, J. S., Appling, A. P., & Karpatne, A. (2020)*. **Physics-Guided Architecture (PGA) of Neural Networks for Quantifying Uncertainty in Lake Temperature Modeling.** In Proceedings of the 2020 SIAM International Conference on Data Mining (SDM) (pp. 532–540). Society for Industrial and Applied Mathematics. [![DOI:10.1137/1.9781611976236.60]](https://doi.org/10.1137/1.9781611976236.60)
